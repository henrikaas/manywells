"""
Tries to find usable wells in a given setof wells.
"""
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from manywells.simulator import SSDFSimulator
from SPSA import configure_wells, create_data_point
from scripts.data_generation.file_utils import save_well_config_and_data
from scripts.data_generation.well import Well

CHOKE_POSITIONS = np.arange(0, 1.0, 0.05)
GAS_LIFT_POSITIONS = np.arange(0.25, 2.0, 0.25)

def is_well_usable(well_and_index):
    i, well = well_and_index
    sim = SSDFSimulator(well.wp, well.bc)

    try: 
        init_x = sim.simulate()
        init_dp = create_data_point(well=well, sim=sim, x=init_x)
        for chk in CHOKE_POSITIONS:
            well.wp.u = chk
            well.bc.w_lg = 0.0
            try:
                dp = sim.simulate()
            except Exception as e:
                print(f"Simulation failed for well {i}: {e}")
                return i, False
            
            if well.has_gas_lift:
                for gl in GAS_LIFT_POSITIONS:
                    well.bc.w_lg = gl

                    try:
                        dp = sim.simulate()
                    except Exception as e:
                        print(f"Simulation failed for well {i}: {e}")
                        return i, False

        return i, init_dp
    except Exception as e:
        print(f"Simulation failed for well {i}: {e}")
        return i, False

def load_well_configs(filename):
    """
    Loads the config data from file
    """
    dataset = pd.read_csv(filename)
    return dataset

# Script
if __name__ == "__main__":
    filepath = 'data/manywells-sol-test-working'
    dataset = load_well_configs(filepath+'/config.csv')
    num_wells = 80

    wells = configure_wells(dataset[:num_wells])
    indexed_wells = list(wells.items())

    working_wells = {}

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(is_well_usable, iw): iw[0] for iw in indexed_wells}
        for future in as_completed(futures):
            i, dp = future.result()
            if dp is not False:
                print(f"Well {i} is usable")
                working_wells[i] = dp
        
    if len(working_wells) > 0:
        print(f"Found {len(working_wells)} usable wells out of {len(wells)} wells")
        for i, dp in working_wells.items():
            save_well_config_and_data(config=wells[i], data=dp, dataset_version=filepath)

        dataset = dataset[num_wells:]
        dataset.to_csv(filepath+'/config.csv', index=False)

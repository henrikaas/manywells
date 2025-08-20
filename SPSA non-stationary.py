"""
Methods for implementing a simple SPSA algorithm
Changes choke and gas lift
"""
import numpy as np
import pandas as pd
from collections import deque
import copy
import os

from manywells.simulator import SSDFSimulator, SimError, WellProperties, BoundaryConditions
from manywells.slip import SlipModel
from manywells.inflow import InflowModel, Vogel
from manywells.choke import ChokeModel, SimpsonChokeModel
from scripts.data_generation.nonstationary_well import NonStationaryWell, NonStationaryBehavior, sample_nonstationary_well
from scripts.data_generation.init_utils import InitGuess
import manywells.pvt as pvt


from scripts.data_generation.file_utils import save_well_config_and_data
from scripts.data_generation.well import Well, sample_well
from constraints import OptimizationProblemProperties

class SPSAOptimization:
    """
    Implementation of SPSA optimization algorithm
    """

    def __init__(self, wells: dict[NonStationaryWell], sigma: float = 0.0):
        self.constr = OptimizationProblemProperties()
        self.wells = wells
        self.n_wells = len(wells)
        self.x_guesses = [deque(maxlen=16)] * self.n_wells # Initiate a FIFOqueue of last guesses for each well

        self.backup_wells = copy.deepcopy(wells) # Back up of wells, if the simulation fails

        # SPSA secific hyperparametres
        # SPSA secific hyperparametres
        self.a = 0.0 # Learning rate controller
        self.b = 0.5 # Dual rate controller
        self.c = 0.25 # Perturbation magnitude controller
        self.A = 50 # Stabilizer
        self.alpha = 0.301 # Decay rate for learning rate
        self.beta = 0.602 # Decay rate for dual rate
        self.gamma = 0.051 # Decay rate for perturbation size
        self.sigma = sigma # Noise level for the oil evaluation
        
        self.lambdas = {'water' : 0.0, 'gl' : 0.0} # Lagrangian multipliers for the constraints

        self.bernoulli = [-1, 1] # Bernoulli directions

    def _draw_directions(self):
        u_direction = np.random.choice(self.bernoulli)
        gl_direction = np.random.choice(self.bernoulli)

        return (u_direction, gl_direction)

    def _sample_variables(self, current_u: float, current_gl: float, ck: float, directions: tuple = None):
        """
        Draws random test directions for u, gl

        :param current_u: current choke for well
        :param current_gl: current gas lift for well
        :param ck: Perturbatiton size
        :directions: Bernoulli directions may be precalculated. Else None
        :return: 2 guesses for u, gl; One positive, one negative
        :return: Bernoulli directions for u, gl
        """

        if directions is None:
            u_direction = np.random.choice(self.bernoulli)
            gl_direction = np.random.choice(self.bernoulli) if self.well.has_gas_lift else 0
        else:
            u_direction = directions[0]
            gl_direction = directions[1]

        guess_pos = [current_u + ck * u_direction,
                     current_gl + ck * gl_direction * self.constr.gl_max]
        guess_neg = [current_u - ck * u_direction,
                     current_gl - ck * gl_direction * self.constr.gl_max]
        
        guess_pos = np.clip(guess_pos, [self.constr.u_min, self.constr.gl_min], [self.constr.u_max, self.constr.gl_max])
        guess_neg = np.clip(guess_neg, [self.constr.u_min, self.constr.gl_min], [self.constr.u_max, self.constr.gl_max])

        print(f"Positive direction: {guess_pos}")
        print(f"Negative direction {guess_neg}")
                
        return [guess_pos, guess_neg], [u_direction, gl_direction]
    
    def _single_SPSA_sim(self, key: int, well: NonStationaryWell, sim: SSDFSimulator, sample_type: str = 'Optimizing'):
        """
        Iterates through one simulation of a well.
        :param well: Well object
        :param sim: Simulator object
        :param sample_type: Type of simulation (Optimizing or Perturbation)
        :return: Data point with simulation results
        """
        n_failed_sims = 0
        if sim.bc.u <= 0.05:
            return self._choked_flow(well, sample_type)
        
        try:
            x = sim.simulate()
            dp = create_data_point(well=well, sim=sim, x=x, decision_type=sample_type)
            return dp, x
        except SimError as e:
            print(f"Simulation failed for well {key} without initial guess. Trying guesses...")
        
        guesses = self.x_guesses[key]
        for i in range(0, len(guesses)):
            
            try:
                sim.x_guess = guesses[i]
                x = sim.simulate()
                dp = create_data_point(well=well, sim=sim, x=x, decision_type=sample_type)
                sim.x_guess = None # Reset the guess

                return dp, x
            except SimError as e:
                print(f"Simulation failed for well {key}. Trying next guess...")
                continue
        
        raise SimError(f"Could not simulate well {key} after {n_failed_sims} attempts. No guesses left.")
    
    def _choked_flow(self, well: NonStationaryWell, sample_type:str):
        """
        If choke == 0, we handle it outside of the simulator
        """
        u = well.bc.u
        w_gl = well.bc.w_lg
        dp = {
        'CHK': u,
        'PBH': 0.0,
        'PWH': 0.0,
        'PDC': well.bc.p_s,
        'TBH': well.bc.T_r,
        'TWH': 0.0,
        'WGL': w_gl,
        'WGAS': 0.0,  # Excluding lift gas
        'WLIQ': 0.0,
        'WOIL': 0.0,
        'WWAT': 0.0,
        'WTOT': 0.0,  # Total mass flow, including lift gas
        'QGL': 0.0,
        'QGAS': 0.0,  # Excluding lift gas
        'QLIQ': 0.0,
        'QOIL': 0.0,
        'QWAT': 0.0,
        'QTOT': 0.0,  # Total volumetric flow, including lift gas
        'FGAS': 0.0,  # Inflow gas mass fraction (WGAS / (WTOT - WGL))
        'FOIL': 0.0,  # Inflow oil mass fraction (WOIL / (WTOT - WGL))
        'FWAT': 0.0,  # Inflow water mass fraction (WWAT / (WTOT - WGL))
        'CHOKED': True,
        'FRBH': None,  # Flow regime at bottomhole
        'FRWH': None,  # Flow regime at wellhead
        }
        if sample_type is not None:
            dp['DCNT'] = sample_type # Type of desicion (exploring, optimizing)

        # Add new data point to dataset
        new_dp = pd.DataFrame(dp, index=[0])  # Passing index since values are scalar
        return new_dp, None
        
    def _save_well_and_config_data(self, well_data: pd.DataFrame, dataset_version: str, k: int):
        path = f"{dataset_version}/iteration_{k}"

        for i, well in enumerate(self.wells.values()):
            save_well_config_and_data(config=well, data=well_data[i], dataset_version=path)

        # Create settings file
        with open(f"data/{path}/settings.txt", 'w') as f:
            f.write(f"iteration: {k}\n")
            f.write(f"lambdas: {self.lambdas}\n\n")
            f.write(f"SPSA hyperparametres:\n")
            f.write(f"a: {self.a}\n")
            f.write(f"b: {self.b}\n")
            f.write(f"c: {self.c}\n")
            f.write(f"A: {self.A}\n")
            f.write(f"alpha: {self.alpha}\n")
            f.write(f"beta: {self.beta}\n")
            f.write(f"gamma: {self.gamma}\n")
            f.write(f"sigma: {self.sigma}\n")
    
    def _do_well_backup(self, well_key: int):
        """
        Back up the well object
        :param well_key: Well key in the dictionary
        :return: Backed up well object
        """
        self.wells[well_key] = self.backup_wells[well_key].copy()
        return self.wells[well_key]
    
    def _handle_simulation_failure(self, well_key: int, sim: SSDFSimulator, u: float = None, gl: float = None):
        # TODO: Important! self.backup_wells is not a truecopy of self.wells, meaning backup_wells is updated together with self.wells
        # Fixed but needs testing
        """
        Handles simulation failure by backing up the well and sampling new conditions
        :param well_idx: Key of well in self.wells
        :param sim: Simulator object
        :return: New well, sim object
        """
        self.wells[well_key] = self._do_well_backup(well_key)

        # well = self.wells[well_key].sample_new_conditions(sample_u=False, sample_w_lg=False)
        well = self.wells[well_key] # Change if we want to sample new conditions
        self.wells[well_key] = well
        sim.wp = well.wp
        sim.bc = well.bc

        if u is not None:
            well.bc.u = u
        if gl is not None:
            well.bc.w_lg = gl

        return well, sim 
        
    def SPSA(self, n_sim: int, dataset_version: str = None, starting_iteration: int = 0):
        """
        Runs an spsa algorithm on multiple wells, trying to find the optimal produciton solution.
        Loosely based on scripts.data_generation.open_loop_stationary.generate_well_data

        :param n_sim: Number of data points to simulate
        """
        
        n_sim_wells = min(self.constr.max_wells, self.n_wells) # Assume we perturb a set of wells (rather than a set of dec. variables)
        well_idxs = np.arange(0, self.n_wells)

        simulators = [SSDFSimulator(w.wp, w.bc) for w in self.wells.values()] # Simulator object for each well
        guesses = InitGuess()
        well_directions = np.zeros((n_sim_wells,2)) # Placeholder for bernoulli directions in each well
        well_deltas = [[0,0]]*n_sim_wells # Placeholder for sample of dec. vars in each well

        # Create data point storage
        well_data = [create_sim_results_df()] * self.n_wells
        pos_well_data = [None] * n_sim_wells # Placeholder for simulation results in positive perturbation
        neg_well_data = [None] * n_sim_wells # Placeholder for simulation results in negative perturbation

        k = starting_iteration +1
        k_prev = starting_iteration
        n_fails = 0
        success = True
        while k <= n_sim:
            if n_fails >= 10:
                raise SimError('Too many fails. Exiting SPSA...')
            # Sample new well conditions
            for key, well in self.wells.items():
                well.update_condtions(k, k_prev)
                simulators[key].wp = well.wp
                simulators[key].bc = well.bc

            chosen_wells_idxs = np.sort(np.random.choice(well_idxs, n_sim_wells, replace=False)) # Randomly choose wells to perturb
            cur_state = np.array([[self.wells[idx].bc.u, self.wells[idx].bc.w_lg] for idx in chosen_wells_idxs]) # Current state of the decision vars in the chosen wells

            # Calculate SPSA parametres
            ak = self.a / ((k + self.A) ** self.alpha)
            bk = self.b / (k**self.beta)
            ck = self.c / (k ** self.gamma)

            # Draw directions (from the Rademacher distribution) for the weels to perturbate
            bern_directions = [self._draw_directions() for _ in range(n_sim_wells)] # Randomly drawn bernoulli directions
            for i, d in enumerate(bern_directions):
                well_directions[i] = d[0], d[1] if self.wells[chosen_wells_idxs[i]].has_gas_lift else 0 # Project back to Theta
                delta, ø = self._sample_variables(current_u=cur_state[i][0],
                                                    current_gl=cur_state[i][1],
                                                    ck=ck, directions=well_directions[i])
                well_deltas[i] = delta

            try:
                # Simulate the perturbed wells
                for i, well_idx in enumerate(chosen_wells_idxs):
                    well = self.wells[well_idx]
                    # Positive direction
                    # Change decision variables
                    well.bc.u = well_deltas[i][0][0]
                    well.bc.w_lg = well_deltas[i][0][1]
                    try:
                        # Simulate positive direction.
                        dp_pos, x = self._single_SPSA_sim(key=well_idx, well=well, sim=simulators[well_idx], sample_type='Perturbation')
                        self.x_guesses[well_idx].append(x)

                    except SimError as e:
                        # If the simulation fails, we back up the well, sample new conditions and try again
                        print(f"Simulation failed for well {i}(L={well.wp.L}), with u: {well.bc.u} & gl: {well.bc.w_lg}.")
                        print("Trying again...")
                        # n_failed_sims += 1

                        # Reset well
                        # well, simulators[well_idx] = self._handle_simulation_failure(well_key=well_idx, sim=simulators[well_idx],
                        #                                                                         u=well_deltas[i][0][0], gl=well_deltas[i][0][1])
                        
                        # Simulate positive direction.
                        dp_pos, x = self._single_SPSA_sim(key=well_idx, well=well, sim=simulators[well_idx], sample_type='Perturbation')
                        self.x_guesses[well_idx].append(x)

                    # Negative direction
                    # Change decision variables
                    well.bc.u = well_deltas[i][1][0]
                    well.bc.w_lg = well_deltas[i][1][1]
                    try:
                        # Simulate negative direction.
                        dp_neg, x = self._single_SPSA_sim(key=well_idx, well=well, sim=simulators[well_idx], sample_type='Perturbation')
                        self.x_guesses[well_idx].append(x)

                    except SimError as e:
                        # If the simulation fails, we back up the well, sample new conditions and try again
                        print(f"Simulation failed for well {i}(L={well.wp.L}), with u: {well.bc.u} & gl: {well.bc.w_lg}.")
                        print("Trying again...")
                        # n_failed_sims += 1
                        
                        # Reset well
                        # well, simulators[well_idx] = self._handle_simulation_failure(well_key=well_idx, sim=simulators[well_idx],
                        #                                                                         u=well_deltas[i][1][0], gl=well_deltas[i][1][1])

                        # Simulate negative direction.
                        dp_neg, x = self._single_SPSA_sim(key=well_idx, well=well, sim=simulators[well_idx], sample_type='Perturbation')
                        self.x_guesses[well_idx].append(x)

                    # Store results
                    pos_well_data[i] = dp_pos
                    neg_well_data[i] = dp_neg
                    well_data[well_idx] = pd.concat([well_data[well_idx], dp_pos, dp_neg], ignore_index=True)

                # Simulate the wells that are not perturbed
                stat_well_data = []
                for i, well in self.wells.items():
                    if i not in chosen_wells_idxs:
                        try:
                            # Simulate the well
                            dp, x = self._single_SPSA_sim(key=i, well=well, sim=simulators[i], sample_type='Stationary')
                            self.x_guesses[i].append(x)
                        except SimError as e:
                            print(f"Simulation failed for well {i}(L={well.wp.L}), trying again...")

                            # Reset well
                            # well, simulators[i] = self._handle_simulation_failure(well_key=i, sim=simulators[i])

                            # Simulate the well
                            dp, x = self._single_SPSA_sim(key=i, well=well, sim=simulators[i], sample_type='Stationary')
                            self.x_guesses[i].append(x)
                        
                        # Store results
                        well_data[i] = pd.concat([well_data[i], dp], ignore_index=True)
                        stat_well_data.append(dp)
                
                # Calculate state in both perturbation directions
                print("Positive state:")
                pos_state = self.calculate_state(well_data=pos_well_data+stat_well_data)
                print("Negative state:")
                neg_state = self.calculate_state(well_data=neg_well_data+stat_well_data)

                # Calculate gradients
                gradient = self._calculate_gradient(pos_state=pos_state, neg_state=neg_state, well_directions=well_directions)
                gradient[:,1] *= self.constr.gl_max # Scale the gradient for gas lift wells

                # Update decision variables
                opt_theta = cur_state - gradient * ak
                opt_theta = np.clip(opt_theta, [self.constr.u_min, self.constr.gl_min], [self.constr.u_max, self.constr.gl_max]) # Project theta back to Theta

                # Run optimizing simulation
                for i, well_idx in enumerate(chosen_wells_idxs):
                    well = self.wells[well_idx]
                    # Change decision variables
                    well.bc.u = opt_theta[i][0]
                    well.bc.w_lg = opt_theta[i][1]

                    try:
                        # Simulate
                        dp, x = self._single_SPSA_sim(key=well_idx, well=well, sim=simulators[well_idx], sample_type='Optimizing')
                        self.x_guesses[well_idx].append(x)

                    except SimError as e:
                        print(f"Simulation failed for well {i}(L={well.wp.L}), with u: {well.bc.u} & gl: {well.bc.w_lg}.")
                        print("Trying again...")
                        # n_failed_sims += 1

                        # Reset well
                        # well, simulators[chosen_wells_idxs[i]] = self._handle_simulation_failure(well_key=well_idx, sim=simulators[well_idx],
                        #                                                                         u=opt_theta[i][0], gl=opt_theta[i][1])

                        # Simulate
                        dp, x = self._single_SPSA_sim(key=well_idx, well=well, sim=simulators[well_idx], sample_type='Optimizing')
                        self.x_guesses[well_idx].append(x)

                    # Store results
                    well_data[well_idx] = pd.concat([well_data[well_idx], dp], ignore_index=True)

                # Calculate the state of the system
                state = self.calculate_state(well_data=well_data)
                success = True

            except SimError as e:
                n_fails += 1
                self.wells = copy.deepcopy(self.backup_wells) # Reset wells to the last successful state
                success = False
                print(f"Simulation failed: {e}")
                print(f"Number of failed simulations: {n_fails}")
                

            if success:
                # Update the Lagrangian multipliers
                self.lambdas['water'] = max(0, self.lambdas['water'] + bk * state['q_water'])
                self.lambdas['gl'] = max(0, self.lambdas['gl'] + bk * state['q_gl'])

                print("--------------------------------------------")
                print(f"Simulation #{k} successful.")
                print(f"SPSA specific parametres: ak: {ak}, bk: {bk}, ck: {ck}")
                print("--------------------------------------------")

                if k % 10 == 0 and dataset_version is not None:
                    print(f"Saving state after {k} successful iterations")
                    self._save_well_and_config_data(well_data=well_data, dataset_version=dataset_version, k=k)

                self.backup_wells = copy.deepcopy(self.wells) # Back up of wells for next iteration
                k_prev = k
                k += 1
                n_fails = 0

        if dataset_version is not None and not os.path.exists(f"data/{dataset_version}/iteration_{n_sim}"):
            self._save_well_and_config_data(well_data=well_data, dataset_version=dataset_version, k=n_sim)

        print("---------------------------------------------")
        print("SPSA successful.")
        print("Final hyperparametres:")
        print(f"Lambdas: {self.lambdas}")
        print(f"ak: {ak}, bk: {bk}, ck: {ck}")

    def calculate_state(self, well_data: list[pd.DataFrame]):
        """
        Calculates the state of the system, in terms of constraints and slack
        Uses the latest simulation for each well
        :param well_data: List of the well data
        :return: Dictionary with the state of the system, Dict{'water': float, 'q_water': float, 'q_gl': float}
        """

        # Total oil production:
        oil = sum([well['WOIL'].values[-1] for well in well_data])
        if self.sigma > 0:
            oil += np.random.normal(0.0, self.sigma) # Add noise to the oil production
        print(f"Total oil produced is {oil}")

        # Total water:
        water = sum([well['WWAT'].values[-1] for well in well_data])
        max_water = self.constr.wat_max
        q_water = water - max_water # Status of constraint q_water
        print(f"Total water produced is {water}, leaving a slack of {-q_water}")

        # Total gas lift:
        gl = sum([well['WGL'].values[-1] for well in well_data])
        max_gl = self.constr.comb_gl_max
        q_gl = gl - max_gl # Status of constraint q_gl
        print(f"Total gas lift used is {gl}, leaving a slack of {-q_gl}")

        return {
            'oil': oil,
            'q_water': q_water,
            'q_gl': q_gl
        }

    def _calculate_gradient(self, pos_state, neg_state, well_directions):
        pos_L = (-pos_state['oil'] + self.lambdas['water'] * pos_state['q_water'] + self.lambdas['gl'] * pos_state['q_gl'])
        neg_L = (-neg_state['oil'] + self.lambdas['water'] * neg_state['q_water'] + self.lambdas['gl'] * neg_state['q_gl'])
        gradient = (pos_L - neg_L).reshape(-1, 1) / (2 * well_directions)
        gradient[np.isinf(gradient)] = 0 # Set infinite values to 0
        return gradient

def load_well_configs():
    """
    Loads the config data from file
    """
    dataset_filename = 'data/testing/SPSA-sol_configs/sample_6.csv'
    dataset = pd.read_csv(dataset_filename)
    return dataset

def configure_wells(config_dataset) -> dict[int, NonStationaryWell]:
    """
    Configures the wells from config file

    :return: Dictionary of wells
    """
    wells = {}
    for index, w in config_dataset.iterrows():
        well_properties = WellProperties(
            L = w['wp.L'],
            D = w['wp.D'],
            rho_l = w['wp.rho_l'],
            R_s = w['wp.R_s'],
            cp_g = w['wp.cp_g'],
            cp_l = w['wp.cp_l'],
            f_D = w['wp.f_D'],
            h = w['wp.h'],
        )

        if w['wp.inflow.class_name'] == 'Vogel':
            inflow = Vogel(
                w_l_max = w['wp.inflow.w_l_max'],
                f_g = w['wp.inflow.f_g']
            )
            well_properties.inflow = inflow
        else:
            print('Inflow Model name not "Vogel"')
        
        if w['wp.choke.class_name'] == 'SimpsonChokeModel':
            choke = SimpsonChokeModel(
                K_c = w['wp.choke.K_c'],
                chk_profile = w['wp.choke.chk_profile']
            )
            well_properties.choke = choke
        else:
            print('Choke Model name not SimpsonChokeModel')

        boundary_conditions = BoundaryConditions(
            p_r = w['bc.p_r'],
            p_s = w['bc.p_s'],
            T_r = w['bc.T_r'],
            T_s = w['bc.T_s'],
            u = w['bc.u'],
            w_lg = w['bc.w_lg']
        )

        gas = pvt.GasProperties(
            name = 'gas',
            R_s = w['gas.R_s'],
            cp = w['gas.cp']
        )

        oil = pvt.LiquidProperties(
            name = 'oil',
            rho = w['oil.rho'],
            cp = w['oil.cp']
        )

        water = pvt.WATER

        f_g = w['fraction.gas']
        f_o = w['fraction.oil']
        f_w = w['fraction.water']

        ns_behavior = NonStationaryBehavior(
            pr_init=w['ns_bhv.pr_init'],
            ps_init=w['ns_bhv.ps_init'],
            init_fractions=['ns_bhv.decay_g'],
        )
        feedback = True

        well = NonStationaryWell(wp=well_properties, bc=boundary_conditions, ns_bhv=ns_behavior,
                    gas=gas, oil=oil, water=water,
                    fractions=(f_g, f_o, f_w),
                    has_gas_lift=w['has_gas_lift'], 
                    feedback=feedback
        )

        wells[index] = well

    return wells

def create_sim_results_df():
    """
    Creates df to store simulation results.
    Based on generate_well_data.simulate_well
    """
    cols = ['CHK', 'PBH', 'PWH', 'PDC', 'TBH', 'TWH',
            'WGL', 'WGAS', 'WLIQ', 'WOIL', 'WWAT', 'WTOT',
            'QGL', 'QGAS', 'QLIQ', 'QOIL', 'QWAT', 'QTOT',
            'FGAS', 'FOIL', 'FWAT', 'CHOKED', 'FRBH', 'FRWH', 'DCNT']
    cols += ['WEEKS']
    well_data = pd.DataFrame(columns=cols, dtype=np.float32)
    well_data['CHOKED'] = well_data['CHOKED'].astype(bool)
    well_data['FRBH'] = well_data['FRBH'].astype(str)
    well_data['FRWH'] = well_data['FRWH'].astype(str)
    well_data['DCNT'] = well_data['DCNT'].astype(str)

    return well_data

def create_data_point(well, sim, x, decision_type=None):
    """
    Creating new data point to add to well_data.
    Based on generate_well_data.simulate_well

    :param well: Well object
    :param sim: Simulation object
    :param x: Simulation to base the data point on.
    :return: New data point, as df
    """

    # Prepare new data point
    df_x = sim.solution_as_df(x)

    df_x['w_g'] = well.wp.A * df_x['alpha'] * df_x['rho_g'] * df_x['v_g']
    df_x['w_l'] = well.wp.A * (1 - df_x['alpha']) * df_x['rho_l'] * df_x['v_l']

    pbh = float(df_x['p'].iloc[0])
    pwh = float(df_x['p'].iloc[-1])
    twh = float(df_x['T'].iloc[-1])
    w_g = float(df_x['w_g'].iloc[-1])  # Including lift gas
    w_l = float(df_x['w_l'].iloc[-1])
    w_tot = w_g + w_l
    w_lg = well.bc.w_lg

    # Get oil and water mass flow rate
    f_g, f_o, f_w = well.fractions
    wlf = f_w / (f_o + f_w)  # Water to liquid fraction
    w_w = w_l * wlf
    w_o = w_l * (1 - wlf)

    # Volumetric flow rates (at standard reference conditions) in Sm³/s
    rho_g = pvt.gas_density(sim.wp.R_s)
    q_g = w_g / rho_g  # Including lift gas
    q_lg = w_lg / rho_g
    q_l = w_l / well.wp.rho_l
    q_o = w_o / well.oil.rho
    q_w = w_w / well.water.rho
    q_tot = q_g + q_l

    # assert abs(q_l - (q_o + q_w)) < 1e-5, f'Liquids do not sum: q_l = {q_l}, q_o + q_w = {q_o + q_w}'

    # Convert volumetric flow rates from Sm³/s to Sm³/h
    SECONDS_PER_HOUR = 3600
    q_g *= SECONDS_PER_HOUR
    q_lg *= SECONDS_PER_HOUR
    q_l *= SECONDS_PER_HOUR
    q_o *= SECONDS_PER_HOUR
    q_w *= SECONDS_PER_HOUR
    q_tot *= SECONDS_PER_HOUR

    # Choked flow?
    choked = well.wp.choke.is_choked(pwh, well.bc.p_s)

    # Flow regime at top and bottom of well
    regime_wh = str(df_x['flow-regime'].iloc[-1])
    regime_bh = str(df_x['flow-regime'].iloc[0])

    # Validate data before adding
    valid_rates = w_l >= 0 and w_g >= 0
    valid_fracs = (0 <= f_g <= 1) and (0 <= f_o <= 1) and (0 <= f_w <= 1)
    if not (valid_rates and valid_fracs):
        raise SimError('Flow rates/mass fractions not valid')  # Count failure - discard simulation
        
    # Discard simulation if total mass flow rate is less than 0.1 kg/s
    if w_l + w_g < 0.1:
        # Simulation did not fail, but solution is invalid (too low flow rate)
        # n_failed_sim += 1  # Count failure - discard simulation
        raise SimError('Total mass flow rate too low')

    # Structure data point in dict
    dp = {
        'CHK': well.bc.u,
        'PBH': pbh,
        'PWH': pwh,
        'PDC': well.bc.p_s,
        'TBH': well.bc.T_r,
        'TWH': twh,
        'WGL': w_lg,
        'WGAS': w_g - w_lg,  # Excluding lift gas
        'WLIQ': w_l,
        'WOIL': w_o,
        'WWAT': w_w,
        'WTOT': w_tot,  # Total mass flow, including lift gas
        'QGL': q_lg,
        'QGAS': q_g - q_lg,  # Excluding lift gas
        'QLIQ': q_l,
        'QOIL': q_o,
        'QWAT': q_w,
        'QTOT': q_tot,  # Total volumetric flow, including lift gas
        'FGAS': f_g,  # Inflow gas mass fraction (WGAS / (WTOT - WGL))
        'FOIL': f_o,  # Inflow oil mass fraction (WOIL / (WTOT - WGL))
        'FWAT': f_w,  # Inflow water mass fraction (WWAT / (WTOT - WGL))
        'CHOKED': choked,
        'FRBH': regime_bh,  # Flow regime at bottomhole
        'FRWH': regime_wh,  # Flow regime at wellhead
    }
    if decision_type is not None:
        dp['DCNT'] = decision_type # Type of desicion (exploring, optimizing)

    # Add new data point to dataset
    new_dp = pd.DataFrame(dp, index=[0])  # Passing index since values are scalar

    return new_dp


if __name__ == '__main__':
    dataset_version = 'testing/testing-spsa-18'

    config_dataset = load_well_configs()

    if os.path.exists(f"data/{dataset_version}"):
        raise FileExistsError(f"Dataset version {dataset_version} already exists. Please choose a different version.")

    for i in range(10):
        wells = configure_wells(config_dataset)
        opt = SPSAOptimization(wells=wells, sigma=0.)
        try:
            opt.SPSA(n_sim=100, dataset_version=f"{dataset_version}/run{i}")
        except SimError as e:
            print(f"Simulation failed: {e}")
            continue
                
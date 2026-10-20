#!/usr/bin/env python3
"""Independent Hamiltonian/Radau audit of the unreleased two-link mechanics pilot.

No canonical module is imported. Kinetic energy comes from Cartesian point-mass
velocities; integration uses (q,p) and cyclic momentum rather than (q,qdot).
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


def kinematic_mass(q):
    a, b = q
    J1 = np.array([[-math.sin(a), 0.], [math.cos(a), 0.]])
    J2 = np.array([[-math.sin(a) - math.sin(a + b), -math.sin(a + b)],
                   [math.cos(a) + math.cos(a + b), math.cos(a + b)]])
    return J1.T @ J1 + J2.T @ J2


def cartesian_energy(state):
    a, b, va, vb = state
    first = np.array([-math.sin(a) * va, math.cos(a) * va])
    second = first + np.array([-math.sin(a + b) * (va + vb), math.cos(a + b) * (va + vb)])
    return float((first @ first + second @ second) / 2)


def inverse_mass(q2):
    c = math.cos(q2)
    return np.array([[1., -1. - c], [-1. - c, 3. + 2. * c]]) / (2. - c*c)


def hamiltonian_flow(q0, v0, torque, duration):
    p0 = kinematic_mass(q0) @ v0
    def rhs(t, qp):
        q, p = qp[:2], qp[2:]
        v = inverse_mass(q[1]) @ p
        # q1 is cyclic. dp2/dt=u2+partial(T)/partial(q2).
        return np.r_[v, torque[0], torque[1] - math.sin(q[1]) * (v[0]**2 + v[0] * v[1])]
    fit = solve_ivp(rhs, [0., duration], np.r_[q0, p0], method='Radau',
                    rtol=2e-12, atol=1e-14, max_step=duration/40)
    assert fit.success
    qp = fit.y[:, -1]
    velocity = inverse_mass(qp[1]) @ qp[2:]
    return np.r_[qp[:2], velocity], float(qp[2] - p0[0] - torque[0] * duration)


def quartic_coefficient(q, torque):
    """q(t)=q0+.5*a*t²+(q4/24)*t⁴+O(t⁶), from Euler-Lagrange."""
    inverse = inverse_mass(q[1])
    acceleration = inverse @ torque
    a, b = acceleration
    s = math.sin(q[1])
    mass_derivative = -s * np.array([[2., 1.], [1., 0.]])
    quadratic_C = s * np.array([-2*a*b - b*b, a*a])
    q4 = -inverse @ (b * mass_derivative @ acceleration + 2 * quadratic_C)
    return acceleration, q4 / 24


def main():
    data = json.loads(Path(__file__).with_name('coupled_context_v14_pilot_results.json').read_text())
    maximum = dict(rest_state=0., energy_state=0., tighter_state=0., coupling_state=0.,
        measurement_RNG=0., energy_work=0., cyclic_momentum=0., stored_scalar=0., mass_kinematics=0.)
    def check(key, a, b):
        maximum[key] = max(maximum[key], float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
    rng = np.random.default_rng(22361020)
    groups = []
    initial_errors, displacement_errors = [], []
    index = 0
    for q2 in np.linspace(-math.pi, math.pi, 17):
        for torque in [( .6, 0.), (0., .6), (.6, -.6)]:
            q0, torque = np.array([.4, q2]), np.array(torque)
            errors = []
            for duration in [.1, .05, .025]:
                row = data['rest_pulses'][index]
                assert row['q0_evaluator_only'] == q0.tolist() and row['command'] == torque.tolist() and row['duration'] == duration
                assert row['v0_prepared_rest'] == [0., 0.]
                state, momentum_error = hamiltonian_flow(q0, np.zeros(2), torque, duration)
                check('rest_state', state, row['final_state_evaluator_only'])
                maximum['cyclic_momentum'] = max(maximum['cyclic_momentum'], abs(momentum_error))
                acceleration, coefficient = quartic_coefficient(q0, torque)
                leading = .5 * duration**2 * acceleration
                displacement = np.array(row['final_state_evaluator_only'])[:2] - q0
                check('stored_scalar', leading, row['leading_response_evaluator_only'])
                check('stored_scalar', displacement, row['displacement_evaluator_only'])
                error = float(np.linalg.norm(displacement - leading))
                errors.append(error)
                check('stored_scalar', error, row['leading_error_norm'])
                e0, e1 = rng.normal(0, 1e-5, 2), rng.normal(0, 1e-5, 2)
                check('measurement_RNG', q0 + e0, row['observed_initial_position'])
                check('measurement_RNG', np.array(row['final_state_evaluator_only'])[:2] + e1, row['observed_final_position'])
                initial_errors.append(e0)
                displacement_errors.append(e1 - e0)
                energy_error = cartesian_energy(row['final_state_evaluator_only']) - torque @ displacement
                check('stored_scalar', energy_error, row['energy_minus_command_work'])
                maximum['energy_work'] = max(maximum['energy_work'], abs(energy_error))
                # Check canonical cyclic momentum as well as independent solver.
                canonical = np.array(row['final_state_evaluator_only'])
                p1 = (kinematic_mass(canonical[:2]) @ canonical[2:])[0]
                maximum['cyclic_momentum'] = max(maximum['cyclic_momentum'], abs(p1 - torque[0] * duration))
                index += 1
            last = data['rest_pulses'][index - 1]
            check('stored_scalar', errors, last['three_duration_error_norms'])
            groups.append(dict(q2=float(q2), command=torque.tolist(), errors=errors,
                quartic_coefficient_norm=float(np.linalg.norm(coefficient)),
                generic=abs(math.sin(q2)) > 1e-12,
                coarse_over_middle=errors[0]/errors[1], middle_over_fine=errors[1]/errors[2]))
    assert index == 153
    for row in data['energy_checks']:
        q, v, torque = rng.uniform(-math.pi, math.pi, 2), rng.uniform(-.6, .6, 2), rng.uniform(-1., 1., 2)
        assert row['q0'] == q.tolist() and row['v0'] == v.tolist() and row['command'] == torque.tolist()
        state, momentum_error = hamiltonian_flow(q, v, torque, .5)
        check('energy_state', state, row['final_state'])
        check('tighter_state', state, row['tighter_state'])
        maximum['cyclic_momentum'] = max(maximum['cyclic_momentum'], abs(momentum_error))
        displacement = np.array(row['final_state'])[:2] - q
        energy_error = cartesian_energy(row['final_state']) - cartesian_energy(np.r_[q, v]) - torque @ displacement
        check('stored_scalar', energy_error, row['energy_work_error'])
        check('stored_scalar', np.max(np.abs(np.array(row['final_state']) - row['tighter_state'])), row['tolerance_state_difference'])
        maximum['energy_work'] = max(maximum['energy_work'], abs(energy_error))
        p1, initial_p1 = (kinematic_mass(np.array(row['final_state'])[:2]) @ np.array(row['final_state'])[2:])[0], (kinematic_mass(q) @ v)[0]
        maximum['cyclic_momentum'] = max(maximum['cyclic_momentum'], abs(p1 - initial_p1 - torque[0] * .5))
    for row in data['coupling_examples']:
        q, torque = np.array([0., row['q2']]), np.array([0., 1.])
        state, momentum_error = hamiltonian_flow(q, np.zeros(2), torque, .05)
        check('coupling_state', state[:2] - q, row['displacement'])
        check('stored_scalar', inverse_mass(q[1]), row['inverse_inertia'])
        check('stored_scalar', np.array(row['displacement'])/(.5 * .05**2), row['short_pulse_response'])
    # Cartesian kinetic Jacobians independently give the declared mass and the
    # exact global eigenvalue lower bound; no sampled-grid positivity inference.
    for q2 in np.linspace(-math.pi, math.pi, 1025):
        mass = kinematic_mass([0., q2])
        c = math.cos(q2)
        check('mass_kinematics', mass, [[3 + 2*c, 1 + c], [1 + c, 1.]])
        assert np.linalg.det(mass) >= 1 - 1e-12
    eigen_lower = 3 - 2 * math.sqrt(2)
    check('stored_scalar', eigen_lower, data['summary']['minimum_inertia_eigenvalue'])
    target_vectors = data['summary']['target_acceleration_vectors']
    # M is affine in cos(q2), so each component's absolute maximum occurs at
    # cos(q2)=+-1. This validates global leading-order torque feasibility.
    torque_max = max(np.max(np.abs(kinematic_mass([0., q2]) @ vector)) for q2 in [0., math.pi] for vector in target_vectors)
    check('stored_scalar', torque_max, data['summary']['maximum_linearized_target_torque'])
    assert torque_max == .5 < 1.
    summary = data['summary']
    assert summary['rest_pulse_episodes'] == 153 and summary['general_energy_episodes'] == 24
    check('stored_scalar', sum(row['duration'] for row in data['rest_pulses']), summary['rest_environment_integration_time'])
    check('stored_scalar', 12., summary['general_environment_integration_time'])
    check('stored_scalar', max(abs(row['energy_work_error']) for row in data['energy_checks']), summary['maximum_energy_work_error'])
    check('stored_scalar', max(row['tolerance_state_difference'] for row in data['energy_checks']), summary['maximum_tighter_solver_state_difference'])
    for expected, actual in zip(groups, summary['generic_pose_error_ratios']):
        assert expected['q2'] == actual['q2'] and expected['command'] == actual['command']
        for key in ['errors', 'coarse_over_middle', 'middle_over_fine']:
            # Ratios at symmetry/roundoff can amplify subtraction-order noise;
            # compare against the exact saved errors, not an independent rate.
            if key != 'errors':
                saved = actual['errors'][0] / actual['errors'][1] if key == 'coarse_over_middle' else actual['errors'][1] / actual['errors'][2]
                check('stored_scalar', saved, actual[key])
    generic = [row for row in groups if row['generic']]
    special = [row for row in groups if not row['generic']]
    assert len(generic) == 42 and len(special) == 9
    assert min(row['quartic_coefficient_norm'] for row in generic) > 0.
    assert max(row['quartic_coefficient_norm'] for row in special) < 1e-12
    assert maximum['rest_state'] < 1e-9 and maximum['energy_state'] < 1e-9 and maximum['tighter_state'] < 1e-9
    assert maximum['coupling_state'] < 1e-11 and maximum['measurement_RNG'] == 0.
    assert maximum['cyclic_momentum'] < 1e-8 and maximum['energy_work'] < 1e-8
    # Stored symmetry remainders are close to floating-point precision; scalar
    # comparison here is for physical quantities, never universal ratio scaling.
    assert maximum['stored_scalar'] < 1e-10
    empirical = np.cov(np.array(initial_errors).T, np.array(displacement_errors).T, ddof=1)[:2, 2:]
    return dict(status='UNRELEASED MECHANICS PILOT INDEPENDENTLY VERIFIED; NO LEARNER COMPARISON',
        independent_representation='Hamiltonian(q,p), Cartesian point-mass kinetic energy, cyclic p1',
        independent_solver='Radau rtol2e-12 atol1e-14 max_step=duration/40',
        rest_pulses_checked=153, general_energy_cases_checked=24, coupling_examples_checked=2,
        complete_unique_episode_count=179, complete_unique_integration_time=21.025,
        canonical_tighter_numerical_reintegrations=24, canonical_total_solver_calls=203,
        maximum_absolute_errors=maximum, exact_global_minimum_mass_eigenvalue=eigen_lower,
        globally_feasible_leading_target_maximum_torque=float(torque_max),
        generic_quartic_groups=42, symmetry_groups=9,
        generic_coarse_middle_ratio_range=[min(row['coarse_over_middle'] for row in generic), max(row['coarse_over_middle'] for row in generic)],
        generic_middle_fine_ratio_range=[min(row['middle_over_fine'] for row in generic), max(row['middle_over_fine'] for row in generic)],
        theoretical_context_displacement_noise_covariance=(-1e-10 * np.eye(2)).tolist(),
        empirical_context_displacement_noise_covariance=empirical.tolist(),
        scope=[
            'Joint-angle coordinates are measured at pulse endpoints; these are not Cartesian end-effector observations.',
            'Shared noisy initial reading enters both context and displacement, giving covariance -R I; no unbiased noisy-context conditional response claim is valid.',
            'Independent context reading removes shared-noise correlation but leaves noisy-context errors-in-variables; joint likelihood or warranted preparation facts still matter.',
            'Naive inverse regression with measured response as regressor has errors-in-variables; no learner/inverse fitted in this pilot.',
            'Rest and constant torque yield even-time expansion with generic quartic remainder; sin(q2)=0 removes quartic term and may give sixth order or exact quadratic special cases.',
            'Symmetry/roundoff error ratios are not evidence against the expansion; the summary field includes all poses despite its generic label.',
            'Global torque feasibility is for leading acceleration targets, not exact finite-pulse endpoint control.',
            'The2 coupling illustrations are extra prescribed episodes beyond153rest+24energy cases; tighter24runs are numerical reintegrations, not independent learning data.',
            'No trained mechanism, noisy response identification result, continuous-feedback controller or general robotics comparison is established.',
        ])


if __name__ == '__main__':
    result = main()
    print(json.dumps(result, indent=2, allow_nan=False))

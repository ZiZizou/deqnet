"""Run all tests sequentially."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all():
    print("#" * 60)
    print("# Running ALL DEQ tests")
    print("#" * 60)

    failed = []
    try:
        from tests.test_deq_solver import (
            test_linear_anderson_matches_solve,
            test_linear_fixed_point_matches_solve,
            test_anderson_unique_attractor,
            test_solve_jacobian_transpose_linear,
            test_check_contraction_positive_margin,
            test_check_contraction_fails_when_gamma_zero,
            test_check_contraction_lambda_min_M_tracks_gamma,
            test_estimate_lipschitz,
        )
        for fn in [test_linear_anderson_matches_solve,
                   test_linear_fixed_point_matches_solve,
                   test_anderson_unique_attractor,
                   test_solve_jacobian_transpose_linear,
                   test_check_contraction_positive_margin,
                   test_check_contraction_fails_when_gamma_zero,
                   test_check_contraction_lambda_min_M_tracks_gamma,
                   test_estimate_lipschitz]:
            try:
                fn()
            except AssertionError as e:
                print(f"  FAIL: {fn.__name__}: {e}")
                failed.append(fn.__name__)
    except Exception as e:
        print(f"  IMPORT FAIL test_deq_solver: {e}")
        failed.append("test_deq_solver import")

    print()

    try:
        from tests.test_devices import (
            test_relu_gain_in_range,
            test_relu_max_slope_matches_gain,
            test_tanh_max_slope_matches_gain,
            test_conductance_max_slope,
            test_device_negation_gain,
            test_relu_forward_shape,
            test_tanh1_translation_invariant,
            test_gain_gradients_flow,
        )
        for fn in [test_relu_gain_in_range,
                   test_relu_max_slope_matches_gain,
                   test_tanh_max_slope_matches_gain,
                   test_conductance_max_slope,
                   test_device_negation_gain,
                   test_relu_forward_shape,
                   test_tanh1_translation_invariant,
                   test_gain_gradients_flow]:
            try:
                fn()
            except AssertionError as e:
                print(f"  FAIL: {fn.__name__}: {e}")
                failed.append(fn.__name__)
    except Exception as e:
        print(f"  IMPORT FAIL test_devices: {e}")
        failed.append("test_devices import")

    print()

    try:
        from tests.test_rhs import (
            test_rhs_zero_at_zero_v_no_input,
            test_rhs_zero_input_unique_equilibrium,
            test_rhs_with_input_at_designated_node,
            test_rhs_linear_matches_explicit,
            test_gamma_is_positive,
            test_rhs_grad_flows_through_gamma,
        )
        for fn in [test_rhs_zero_at_zero_v_no_input,
                   test_rhs_zero_input_unique_equilibrium,
                   test_rhs_with_input_at_designated_node,
                   test_rhs_linear_matches_explicit,
                   test_gamma_is_positive,
                   test_rhs_grad_flows_through_gamma]:
            try:
                fn()
            except AssertionError as e:
                print(f"  FAIL: {fn.__name__}: {e}")
                failed.append(fn.__name__)
    except Exception as e:
        print(f"  IMPORT FAIL test_rhs: {e}")
        failed.append("test_rhs import")

    print()

    try:
        from tests.test_equilibrium_solve import (
            test_forward_solves_equilibrium,
            test_implicit_grad_matches_unrolled,
            test_phantom_backward_does_not_error,
            test_gradcheck_implicit_vs_unrolled,
            test_multi_layer_grad_flows_to_earlier_layers,
            test_linear_solve_layer_gradient_flow,
            test_linear_solve_layer_matches_direct_solve,
        )
        for fn in [test_forward_solves_equilibrium,
                   test_implicit_grad_matches_unrolled,
                   test_phantom_backward_does_not_error,
                   test_gradcheck_implicit_vs_unrolled,
                   test_multi_layer_grad_flows_to_earlier_layers,
                   test_linear_solve_layer_gradient_flow,
                   test_linear_solve_layer_matches_direct_solve]:
            try:
                fn()
            except AssertionError as e:
                print(f"  FAIL: {fn.__name__}: {e}")
                failed.append(fn.__name__)
    except Exception as e:
        print(f"  IMPORT FAIL test_equilibrium_solve: {e}")
        failed.append("test_equilibrium_solve import")

    print()

    try:
        from tests.test_deq_end_to_end import (
            test_deq_forward_shape,
            test_deq_backward_and_step,
            test_deq_solver_stats_logged,
        )
        for fn in [test_deq_forward_shape,
                   test_deq_backward_and_step,
                   test_deq_solver_stats_logged]:
            try:
                fn()
            except AssertionError as e:
                print(f"  FAIL: {fn.__name__}: {e}")
                failed.append(fn.__name__)
    except Exception as e:
        print(f"  IMPORT FAIL test_deq_end_to_end: {e}")
        failed.append("test_deq_end_to_end import")

    print()
    if failed:
        print(f"!!! {len(failed)} tests failed:")
        for n in failed:
            print(f"  - {n}")
        sys.exit(1)
    print("ALL TESTS PASSED.")


if __name__ == '__main__':
    run_all()

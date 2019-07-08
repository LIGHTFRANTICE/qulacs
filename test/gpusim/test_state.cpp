#include <gtest/gtest.h>
#include "../util/util.h"
#include <cppsim/state.hpp>
#include <cppsim/utility.hpp>
#include <cppsim/state_gpu.hpp>

// post-selection probability check
// referred to test/csim/test_stat.cpp
TEST(StatOperationTest, ProbTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;
	const double eps = 1e-14;

	//CTYPE* state = allocate_quantum_state(dim);
	QuantumStateGpu state(n);
	Eigen::MatrixXcd P0(2, 2), P1(2, 2);
	P0 << 1, 0, 0, 0;
	P1 << 0, 0, 0, 1;

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		state.set_Haar_random_state();
		ASSERT_NEAR(state_norm_host(state.data(), dim), 1, eps);
		auto state_cpp = state.data_cpp();
		Eigen::VectorXcd test_state(dim);
		for (ITYPE i = 0; i < dim; ++i) test_state[i] = state_cpp[i];

		for (UINT target = 0; target < n; ++target) {
			double p0 = M0_prob_host(target, state.data(), dim);
			double p1 = M1_prob_host(target, state.data(), dim);
			ASSERT_NEAR((get_expanded_eigen_matrix_with_identity(target, P0, n)*test_state).squaredNorm(), p0, eps);
			ASSERT_NEAR((get_expanded_eigen_matrix_with_identity(target, P1, n)*test_state).squaredNorm(), p1, eps);
			ASSERT_NEAR(p0 + p1, 1, eps);
		}
	}
}

// marginal probability check
// referred to test/csim/test_stat.cpp
TEST(StatOperationTest, MarginalProbTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;
	const double eps = 1e-14;

	QuantumStateGpu state(n);
	Eigen::MatrixXcd P0(2, 2), P1(2, 2), Identity(2, 2);
	P0 << 1, 0, 0, 0;
	P1 << 0, 0, 0, 1;
	Identity << 1, 0, 0, 1;

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		state.set_Haar_random_state();
		ASSERT_NEAR(state_norm_host(state.data(), dim), 1, eps);
		auto state_cpp = state.data_cpp();
		Eigen::VectorXcd test_state(dim);
		for (ITYPE i = 0; i < dim; ++i) test_state[i] = state_cpp[i];

		for (UINT target = 0; target < n; ++target) {
			// merginal probability check
			Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
			std::vector<UINT> index_list, measured_value_list;

			index_list.clear();
			measured_value_list.clear();
			for (UINT i = 0; i < n; ++i) {
				UINT measured_value = rand_int(3);
				if (measured_value != 2) {
					measured_value_list.push_back(measured_value);
					index_list.push_back(i);
				}
				if (measured_value == 0) {
					mat = kronecker_product(P0, mat);
				}
				else if (measured_value == 1) {
					mat = kronecker_product(P1, mat);
				}
				else {
					mat = kronecker_product(Identity, mat);
				}
			}
			double test_marginal_prob = (mat*test_state).squaredNorm();
			double res = marginal_prob_host(index_list.data(), measured_value_list.data(), (UINT)index_list.size(), state.data(), dim);
			ASSERT_NEAR(test_marginal_prob, res, eps);
		}
	}
}

// entropy
TEST(StatOperationTest, EntropyTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;
	const double eps = 1e-14;

	// CTYPE* state = allocate_quantum_state(dim);
	void* state = allocate_quantum_state_host(dim);
	CTYPE* state_cpp = (CTYPE*)malloc(dim * sizeof(CTYPE));

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		initialize_Haar_random_state_host(state, dim);
		ASSERT_NEAR(state_norm_host(state, dim), 1, eps);
		Eigen::VectorXcd test_state(dim);
		get_quantum_state_host(state, state_cpp, dim);
		for (ITYPE i = 0; i < dim; ++i) test_state[i] = state_cpp[i];

		for (UINT target = 0; target < n; ++target) {
			double ent = 0;
			for (ITYPE ind = 0; ind < dim; ++ind) {
				double prob = norm(test_state[ind]);
				if (prob > eps)
					ent += -prob * log(prob);
			}
			ASSERT_NEAR(ent, measurement_distribution_entropy_host(state, dim), eps);
		}
	}
	free(state_cpp);
	release_quantum_state_host(state);
}

// inner product
TEST(StatOperationTest, InnerProductTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;
	const double eps = 1e-14;

	CTYPE* state = allocate_quantum_state(dim);
	void* state_gpu = allocate_quantum_state_host(dim);
	CTYPE* buffer = allocate_quantum_state(dim);
	void* buffer_gpu = allocate_quantum_state_host(dim);
	for (UINT rep = 0; rep < max_repeat; ++rep) {

		initialize_Haar_random_state_host(state_gpu, dim);
		ASSERT_NEAR(state_norm_host(state_gpu, dim), 1, eps);
		Eigen::VectorXcd test_state(dim);
		get_quantum_state_host(state_gpu, state, dim);
		for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

		for (UINT target = 0; target < n; ++target) {
			initialize_Haar_random_state_host(buffer_gpu, dim);
			get_quantum_state_host(buffer_gpu, buffer, dim);
			CTYPE inp = inner_product_host(buffer_gpu, state_gpu, dim);
			Eigen::VectorXcd test_buffer(dim);
			for (ITYPE i = 0; i < dim; ++i) test_buffer[i] = buffer[i];
			std::complex<double> test_inp = (test_buffer.adjoint() * test_state);
			ASSERT_NEAR(creal(inp), test_inp.real(), eps);
			ASSERT_NEAR(cimag(inp), test_inp.imag(), eps);
		}
	}
	release_quantum_state(state);
	release_quantum_state_host(state_gpu);
	release_quantum_state(buffer);
	release_quantum_state_host(buffer_gpu);
}


// single qubit expectation value
TEST(StatOperationTest, SingleQubitExpectationValueTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;
	const double eps = 1e-14;

	CTYPE* state = allocate_quantum_state(dim);
	void* state_gpu = allocate_quantum_state_host(dim);
	Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);
	Eigen::MatrixXcd pauli_op;
	Identity << 1, 0, 0, 1;
	X << 0, 1, 1, 0;
	Z << 1, 0, 0, -1;
	Y << 0, -1.i, 1.i, 0;

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		initialize_Haar_random_state_host(state_gpu, dim);
		ASSERT_NEAR(state_norm_host(state_gpu, dim), 1, eps);
		Eigen::VectorXcd test_state(dim);
		get_quantum_state_host(state_gpu, state, dim);
		for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

		for (UINT target = 0; target < n; ++target) {
			// single qubit expectation value check
			target = rand_int(n);
			UINT pauli = rand_int(3) + 1;
			if (pauli == 0) pauli_op = Identity;
			else if (pauli == 1) pauli_op = X;
			else if (pauli == 2) pauli_op = Y;
			else if (pauli == 3) pauli_op = Z;
			std::complex<double> value = (test_state.adjoint()*get_expanded_eigen_matrix_with_identity(target, pauli_op, n)*test_state);
			ASSERT_NEAR(value.imag(), 0, eps);
			double test_expectation = value.real();
			double expectation = expectation_value_single_qubit_Pauli_operator_host(pauli, target, state_gpu, dim);
			ASSERT_NEAR(expectation, test_expectation, eps);
		}
	}
	release_quantum_state(state);
	release_quantum_state_host(state_gpu);
}

// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitExpectationValueWholeTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;
	const double eps = 1e-14;

	CTYPE* state = allocate_quantum_state(dim);
	void* state_gpu = allocate_quantum_state_host(dim);
	Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);
	Eigen::MatrixXcd pauli_op;
	Identity << 1, 0, 0, 1;
	X << 0, 1, 1, 0;
	Z << 1, 0, 0, -1;
	Y << 0, -1.i, 1.i, 0;

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		initialize_Haar_random_state_host(state_gpu, dim);
		ASSERT_NEAR(state_norm_host(state_gpu, dim), 1, eps);
		Eigen::VectorXcd test_state(dim);
		get_quantum_state_host(state_gpu, state, dim);
		for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

		for (UINT target = 0; target < n; ++target) {
			// multi qubit expectation whole list value check
			Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
			std::vector<UINT> pauli_whole;
			for (UINT i = 0; i < n; ++i) {
				UINT pauli = rand_int(4);
				if (pauli == 0) pauli_op = Identity;
				else if (pauli == 1) pauli_op = X;
				else if (pauli == 2) pauli_op = Y;
				else if (pauli == 3) pauli_op = Z;
				mat = kronecker_product(pauli_op, mat);
				pauli_whole.push_back(pauli);
			}
			std::complex<double> value = (test_state.adjoint()*mat*test_state);
			ASSERT_NEAR(value.imag(), 0, eps);
			double test_expectation = value.real();
			double expectation = expectation_value_multi_qubit_Pauli_operator_whole_list_host(pauli_whole.data(), n, state_gpu, dim);
			ASSERT_NEAR(expectation, test_expectation, eps);
		}
	}
	release_quantum_state(state);
	release_quantum_state_host(state_gpu);
}

// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitExpectationValueZopWholeTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;
	const double eps = 1e-14;

	CTYPE* state = allocate_quantum_state(dim);
	void* state_gpu = allocate_quantum_state_host(dim);
	Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);
	Eigen::MatrixXcd pauli_op;
	Identity << 1, 0, 0, 1;
	X << 0, 1, 1, 0;
	Z << 1, 0, 0, -1;
	Y << 0, -1.i, 1.i, 0;

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		initialize_Haar_random_state_host(state_gpu, dim);
		ASSERT_NEAR(state_norm_host(state_gpu, dim), 1, eps);
		Eigen::VectorXcd test_state(dim);
		get_quantum_state_host(state_gpu, state, dim);
		for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

		for (UINT target = 0; target < n; ++target) {
			// multi qubit expectation whole list value check
			Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
			std::vector<UINT> pauli_whole;
			for (UINT i = 0; i < n; ++i) {
				UINT pauli = rand_int(2);
				if (pauli == 1) pauli = 3;
				if (pauli == 0) pauli_op = Identity;
				else pauli_op = Z;
				mat = kronecker_product(pauli_op, mat);
				pauli_whole.push_back(pauli);
			}
			std::complex<double> value = (test_state.adjoint()*mat*test_state);
			ASSERT_NEAR(value.imag(), 0, eps);
			double test_expectation = value.real();
			double expectation = expectation_value_multi_qubit_Pauli_operator_whole_list_host(pauli_whole.data(), n, state_gpu, dim);
			ASSERT_NEAR(expectation, test_expectation, eps);
		}
	}
	release_quantum_state(state);
	release_quantum_state_host(state_gpu);
}


// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitExpectationValuePartialTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const double eps = 1e-14;
	const UINT max_repeat = 10;

	CTYPE* state = allocate_quantum_state(dim);
	void* state_gpu = allocate_quantum_state_host(dim);
	for (UINT rep = 0; rep < max_repeat; ++rep) {
		initialize_Haar_random_state_host(state_gpu, dim);
		ASSERT_NEAR(state_norm_host(state_gpu, dim), 1, eps);
		Eigen::VectorXcd test_state(dim);
		get_quantum_state_host(state_gpu, state, dim);
		for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

		Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);
		Identity << 1, 0, 0, 1;
		X << 0, 1, 1, 0;
		Z << 1, 0, 0, -1;
		Y << 0, -1.i, 1.i, 0;

		for (UINT target = 0; target < n; ++target) {
			// multi qubit expectation partial list value check
			Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
			Eigen::MatrixXcd pauli_op;

			std::vector<UINT> pauli_partial, pauli_index;
			std::vector<std::pair<UINT, UINT>> pauli_partial_pair;
			for (UINT i = 0; i < n; ++i) {
				UINT pauli = rand_int(4);
				if (pauli == 0) pauli_op = Identity;
				else if (pauli == 1) pauli_op = X;
				else if (pauli == 2) pauli_op = Y;
				else if (pauli == 3) pauli_op = Z;
				mat = kronecker_product(pauli_op, mat);
				if (pauli != 0) {
					pauli_partial_pair.push_back(std::make_pair(i, pauli));
				}
			}
			std::random_shuffle(pauli_partial_pair.begin(), pauli_partial_pair.end());
			for (auto val : pauli_partial_pair) {
				pauli_index.push_back(val.first);
				pauli_partial.push_back(val.second);
			}
			std::complex<double> value = (test_state.adjoint()*mat*test_state);
			ASSERT_NEAR(value.imag(), 0, eps);
			double test_expectation = value.real();
			double expectation = expectation_value_multi_qubit_Pauli_operator_partial_list_host(pauli_index.data(), pauli_partial.data(), (UINT)pauli_index.size(), state_gpu, dim);
			ASSERT_NEAR(expectation, test_expectation, eps);
		}
	}
	release_quantum_state(state);
	release_quantum_state_host(state_gpu);
}

// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitExpectationValueZopPartialTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const double eps = 1e-14;
	const UINT max_repeat = 10;

	CTYPE* state = allocate_quantum_state(dim);
	void* state_gpu = allocate_quantum_state_host(dim);
	for (UINT rep = 0; rep < max_repeat; ++rep) {
		initialize_Haar_random_state_host(state_gpu, dim);
		ASSERT_NEAR(state_norm_host(state_gpu, dim), 1, eps);
		Eigen::VectorXcd test_state(dim);
		get_quantum_state_host(state_gpu, state, dim);
		for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

		Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);
		Identity << 1, 0, 0, 1;
		X << 0, 1, 1, 0;
		Z << 1, 0, 0, -1;
		Y << 0, -1.i, 1.i, 0;

		for (UINT target = 0; target < n; ++target) {
			// multi qubit expectation partial list value check
			Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
			Eigen::MatrixXcd pauli_op;

			std::vector<UINT> pauli_partial, pauli_index;
			std::vector<std::pair<UINT, UINT>> pauli_partial_pair;
			for (UINT i = 0; i < n; ++i) {
				UINT pauli = rand_int(2);
				if (pauli == 1) pauli = 3;
				if (pauli == 0) pauli_op = Identity;
				else pauli_op = Z;
				mat = kronecker_product(pauli_op, mat);
				if (pauli != 0) {
					pauli_partial_pair.push_back(std::make_pair(i, pauli));
				}
			}
			std::random_shuffle(pauli_partial_pair.begin(), pauli_partial_pair.end());
			for (auto val : pauli_partial_pair) {
				pauli_index.push_back(val.first);
				pauli_partial.push_back(val.second);
			}
			std::complex<double> value = (test_state.adjoint()*mat*test_state);
			ASSERT_NEAR(value.imag(), 0, eps);
			double test_expectation = value.real();
			double expectation = expectation_value_multi_qubit_Pauli_operator_partial_list_host(pauli_index.data(), pauli_partial.data(), (UINT)pauli_index.size(), state_gpu, dim);
			ASSERT_NEAR(expectation, test_expectation, eps);
		}
	}
	release_quantum_state(state);
	release_quantum_state_host(state_gpu);
}

TEST(StatOperationTest, MultiQubitTransitionAmplitudeWholeTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;
	const double eps = 1e-14;

	CTYPE* state_ket = allocate_quantum_state(dim);
	CTYPE* state_bra = allocate_quantum_state(dim);
	void* state_ket_gpu = allocate_quantum_state_host(dim);
	void* state_bra_gpu = allocate_quantum_state_host(dim);

	Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);
	Eigen::MatrixXcd pauli_op;
	Identity << 1, 0, 0, 1;
	X << 0, 1, 1, 0;
	Z << 1, 0, 0, -1;
	Y << 0, -1.i, 1.i, 0;

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		initialize_Haar_random_state_host(state_ket_gpu, dim);
		initialize_Haar_random_state_host(state_bra_gpu, dim);
		ASSERT_NEAR(state_norm_host(state_ket_gpu, dim), 1, eps);
		ASSERT_NEAR(state_norm_host(state_bra_gpu, dim), 1, eps);

		Eigen::VectorXcd test_state_ket(dim);
		Eigen::VectorXcd test_state_bra(dim);
		get_quantum_state_host(state_ket_gpu, state_ket, dim);
		get_quantum_state_host(state_bra_gpu, state_bra, dim);

		for (ITYPE i = 0; i < dim; ++i) test_state_ket[i] = state_ket[i];
		for (ITYPE i = 0; i < dim; ++i) test_state_bra[i] = state_bra[i];

		for (UINT target = 0; target < n; ++target) {
			// multi qubit expectation whole list value check
			Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
			std::vector<UINT> pauli_whole;
			for (UINT i = 0; i < n; ++i) {
				UINT pauli = rand_int(4);
				if (pauli == 0) pauli_op = Identity;
				else if (pauli == 1) pauli_op = X;
				else if (pauli == 2) pauli_op = Y;
				else if (pauli == 3) pauli_op = Z;
				mat = kronecker_product(pauli_op, mat);
				pauli_whole.push_back(pauli);
			}
			std::complex<double> test_transition_amplitude = (test_state_bra.adjoint()*mat*test_state_ket);
			CPPCTYPE transition_amplitude = transition_amplitude_multi_qubit_Pauli_operator_whole_list_host(pauli_whole.data(), n, state_bra_gpu, state_ket_gpu, dim);
			ASSERT_NEAR(transition_amplitude.real(), test_transition_amplitude.real(), eps);
			ASSERT_NEAR(transition_amplitude.imag(), test_transition_amplitude.imag(), eps);
		}
	}
	release_quantum_state(state_ket);
	release_quantum_state(state_bra);
	release_quantum_state_host(state_ket_gpu);
	release_quantum_state_host(state_bra_gpu);
}

// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitTransitionAmplitudeZopWholeTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;
	const double eps = 1e-14;

	CTYPE* state_ket = allocate_quantum_state(dim);
	CTYPE* state_bra = allocate_quantum_state(dim);
	void* state_ket_gpu = allocate_quantum_state_host(dim);
	void* state_bra_gpu = allocate_quantum_state_host(dim);

	Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);
	Eigen::MatrixXcd pauli_op;
	Identity << 1, 0, 0, 1;
	X << 0, 1, 1, 0;
	Z << 1, 0, 0, -1;
	Y << 0, -1.i, 1.i, 0;

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		initialize_Haar_random_state_host(state_ket_gpu, dim);
		initialize_Haar_random_state_host(state_bra_gpu, dim);
		ASSERT_NEAR(state_norm_host(state_ket_gpu, dim), 1, eps);
		ASSERT_NEAR(state_norm_host(state_bra_gpu, dim), 1, eps);

		Eigen::VectorXcd test_state_ket(dim);
		Eigen::VectorXcd test_state_bra(dim);
		get_quantum_state_host(state_ket_gpu, state_ket, dim);
		get_quantum_state_host(state_bra_gpu, state_bra, dim);

		for (ITYPE i = 0; i < dim; ++i) test_state_ket[i] = state_ket[i];
		for (ITYPE i = 0; i < dim; ++i) test_state_bra[i] = state_bra[i];

		for (UINT target = 0; target < n; ++target) {
			// multi qubit expectation whole list value check
			Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
			std::vector<UINT> pauli_whole;
			for (UINT i = 0; i < n; ++i) {
				UINT pauli = rand_int(2);
				if (pauli == 1) pauli = 3;
				if (pauli == 0) pauli_op = Identity;
				else pauli_op = Z;
				mat = kronecker_product(pauli_op, mat);
				pauli_whole.push_back(pauli);
			}
			std::complex<double> test_transition_amplitude = (test_state_bra.adjoint()*mat*test_state_ket);
			CPPCTYPE transition_amplitude = transition_amplitude_multi_qubit_Pauli_operator_whole_list_host(pauli_whole.data(), n, state_bra_gpu, state_ket_gpu, dim);
			ASSERT_NEAR(transition_amplitude.real(), test_transition_amplitude.real(), eps);
			ASSERT_NEAR(transition_amplitude.imag(), test_transition_amplitude.imag(), eps);
		}
	}
	release_quantum_state(state_ket);
	release_quantum_state(state_bra);
	release_quantum_state_host(state_ket_gpu);
	release_quantum_state_host(state_bra_gpu);
}

// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitTransitionAmplitudePartialTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;
	const double eps = 1e-14;

	CTYPE* state_ket = allocate_quantum_state(dim);
	CTYPE* state_bra = allocate_quantum_state(dim);
	void* state_ket_gpu = allocate_quantum_state_host(dim);
	void* state_bra_gpu = allocate_quantum_state_host(dim);

	Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);
	Eigen::MatrixXcd pauli_op;
	Identity << 1, 0, 0, 1;
	X << 0, 1, 1, 0;
	Z << 1, 0, 0, -1;
	Y << 0, -1.i, 1.i, 0;

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		initialize_Haar_random_state_host(state_ket_gpu, dim);
		initialize_Haar_random_state_host(state_bra_gpu , dim);
		ASSERT_NEAR(state_norm_host(state_ket_gpu, dim), 1, eps);
		ASSERT_NEAR(state_norm_host(state_bra_gpu, dim), 1, eps);

		Eigen::VectorXcd test_state_ket(dim);
		Eigen::VectorXcd test_state_bra(dim);
		get_quantum_state_host(state_ket_gpu, state_ket, dim);
		get_quantum_state_host(state_bra_gpu, state_bra, dim);

		for (ITYPE i = 0; i < dim; ++i) test_state_ket[i] = state_ket[i];
		for (ITYPE i = 0; i < dim; ++i) test_state_bra[i] = state_bra[i];

		for (UINT target = 0; target < n; ++target) {
			// multi qubit expectation partial list value check
			Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
			Eigen::MatrixXcd pauli_op;

			std::vector<UINT> pauli_partial, pauli_index;
			std::vector<std::pair<UINT, UINT>> pauli_partial_pair;
			for (UINT i = 0; i < n; ++i) {
				UINT pauli = rand_int(4);
				if (pauli == 0) pauli_op = Identity;
				else if (pauli == 1) pauli_op = X;
				else if (pauli == 2) pauli_op = Y;
				else if (pauli == 3) pauli_op = Z;
				mat = kronecker_product(pauli_op, mat);
				if (pauli != 0) {
					pauli_partial_pair.push_back(std::make_pair(i, pauli));
				}
			}
			std::random_shuffle(pauli_partial_pair.begin(), pauli_partial_pair.end());
			for (auto val : pauli_partial_pair) {
				pauli_index.push_back(val.first);
				pauli_partial.push_back(val.second);
			}
			std::complex<double> test_transition_amplitude = (test_state_bra.adjoint()*mat*test_state_ket);
			CPPCTYPE transition_amplitude = transition_amplitude_multi_qubit_Pauli_operator_partial_list_host(pauli_index.data(), pauli_partial.data(), (UINT)pauli_index.size(), state_bra_gpu, state_ket_gpu, dim);
			ASSERT_NEAR(transition_amplitude.real(), test_transition_amplitude.real(), eps);
			ASSERT_NEAR(transition_amplitude.imag(), test_transition_amplitude.imag(), eps);
		}
	}
	release_quantum_state(state_ket);
	release_quantum_state(state_bra);
	release_quantum_state_host(state_ket_gpu);
	release_quantum_state_host(state_bra_gpu);
}

// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitTransitionAmplitudeZopPartialTest) {
	const UINT n = 6;
	const ITYPE dim = 1ULL << n;
	const UINT max_repeat = 10;
	const double eps = 1e-14;

	CTYPE* state_ket = allocate_quantum_state(dim);
	CTYPE* state_bra = allocate_quantum_state(dim);
	void* state_ket_gpu = allocate_quantum_state_host(dim);
	void* state_bra_gpu = allocate_quantum_state_host(dim);

	Eigen::MatrixXcd Identity(2, 2), X(2, 2), Y(2, 2), Z(2, 2);
	Eigen::MatrixXcd pauli_op;
	Identity << 1, 0, 0, 1;
	X << 0, 1, 1, 0;
	Z << 1, 0, 0, -1;
	Y << 0, -1.i, 1.i, 0;

	for (UINT rep = 0; rep < max_repeat; ++rep) {
		initialize_Haar_random_state_host(state_ket_gpu, dim);
		initialize_Haar_random_state_host(state_bra_gpu, dim);
		ASSERT_NEAR(state_norm_host(state_ket_gpu, dim), 1, eps);
		ASSERT_NEAR(state_norm_host(state_bra_gpu, dim), 1, eps);

		Eigen::VectorXcd test_state_ket(dim);
		Eigen::VectorXcd test_state_bra(dim);
		get_quantum_state_host(state_ket_gpu, state_ket, dim);
		get_quantum_state_host(state_bra_gpu, state_bra, dim);

		for (ITYPE i = 0; i < dim; ++i) test_state_ket[i] = state_ket[i];
		for (ITYPE i = 0; i < dim; ++i) test_state_bra[i] = state_bra[i];

		for (UINT target = 0; target < n; ++target) {
			// multi qubit expectation partial list value check
			Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
			Eigen::MatrixXcd pauli_op;

			std::vector<UINT> pauli_partial, pauli_index;
			std::vector<std::pair<UINT, UINT>> pauli_partial_pair;
			for (UINT i = 0; i < n; ++i) {
				UINT pauli = rand_int(2);
				if (pauli == 1) pauli = 3;
				if (pauli == 0) pauli_op = Identity;
				else pauli_op = Z;
				mat = kronecker_product(pauli_op, mat);
				if (pauli != 0) {
					pauli_partial_pair.push_back(std::make_pair(i, pauli));
				}
			}
			std::random_shuffle(pauli_partial_pair.begin(), pauli_partial_pair.end());
			for (auto val : pauli_partial_pair) {
				pauli_index.push_back(val.first);
				pauli_partial.push_back(val.second);
			}
			std::complex<double> test_transition_amplitude = (test_state_bra.adjoint()*mat*test_state_ket);
			CPPCTYPE transition_amplitude = transition_amplitude_multi_qubit_Pauli_operator_partial_list_host(pauli_index.data(), pauli_partial.data(), (UINT)pauli_index.size(), state_bra_gpu, state_ket_gpu, dim);
			ASSERT_NEAR(transition_amplitude.real(), test_transition_amplitude.real(), eps);
			ASSERT_NEAR(transition_amplitude.imag(), test_transition_amplitude.imag(), eps);
		}
	}
	release_quantum_state(state_ket);
	release_quantum_state(state_bra);
	release_quantum_state_host(state_ket_gpu);
	release_quantum_state_host(state_bra_gpu);
}

TEST(StateTest, GenerateAndRelease) {    
    UINT n = 10;
    double eps = 1e-14;
    QuantumState state(n);
    ASSERT_EQ(state.qubit_count, n);
    ASSERT_EQ(state.dim, 1ULL << n);
    state.set_zero_state();
    for (UINT i = 0; i < state.dim; ++i) {
        if (i == 0) ASSERT_NEAR(abs(state.data_cpp()[i] - 1.), 0, eps);
        else ASSERT_NEAR(abs(state.data_cpp()[i]), 0, eps);
    }
    Random random;
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        ITYPE basis = random.int64()%state.dim;
        state.set_computational_basis(basis);
        for (UINT i = 0; i < state.dim; ++i) {
            if (i == basis) ASSERT_NEAR(abs(state.data_cpp()[i] - 1.), 0, eps);
            else ASSERT_NEAR(abs(state.data_cpp()[i]), 0, eps);
        }
    }
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        state.set_Haar_random_state();
        ASSERT_NEAR(state.get_norm(),1.,eps);
    }
}

TEST(StateTest, Sampling) {
    UINT n = 10;
    QuantumState state(n);
    state.set_Haar_random_state();
    state.set_computational_basis(100);
    auto res1 = state.sampling(1024);
    state.set_computational_basis(100);
    auto res2 = state.sampling(1024);
}


TEST(StateTest, SetState) {
	const double eps = 1e-10;
	const UINT n = 10;
	QuantumStateGpu state(n);
	const ITYPE dim = 1ULL << n;
	std::vector<std::complex<double>> state_vector(dim);
	for (UINT i = 0; i < dim; ++i) {
		double d = (double)i;
		state_vector[i] = d + std::complex<double>(0, 1)*(d + 0.1);
	}
	state.load(state_vector);
	for (UINT i = 0; i < dim; ++i) {
		ASSERT_NEAR(state.data_cpp()[i].real(), state_vector[i].real(), eps);
		ASSERT_NEAR(state.data_cpp()[i].imag(), state_vector[i].imag(), eps);
	}
}

TEST(StateTest, CopyState) {
	const double eps = 1e-10;
	const UINT n = 7;
	QuantumState state(n);
	state.set_Haar_random_state();
	const ITYPE dim = 1ULL << n;
	void* state_gpu = allocate_quantum_state_host(dim);
	copy_quantum_state_from_host_to_device(state_gpu, state.data(), dim);
	CPPCTYPE* state_cpu_copy = (CPPCTYPE*)malloc(sizeof(CPPCTYPE)*dim);
	get_quantum_state_host(state_gpu, state_cpu_copy, dim);

	for (UINT i = 0; i < dim; ++i) {
		ASSERT_NEAR(state.data_cpp()[i].real(), state_cpu_copy[i].real(), eps);
		ASSERT_NEAR(state.data_cpp()[i].imag(), state_cpu_copy[i].imag(), eps);
	}
	release_quantum_state_host(state_gpu);
}

TEST(StateTest, CopyStateDevice) {
	const double eps = 1e-10;
	const UINT n = 8;
	const ITYPE dim = 1ULL << n;
	void* state1_gpu = allocate_quantum_state_host(dim);
	void* state2_gpu = allocate_quantum_state_host(dim);

	initialize_Haar_random_state_host(state1_gpu, dim);

	CPPCTYPE* state1_cpu_copy = (CPPCTYPE*)malloc(sizeof(CPPCTYPE)*dim);
	CPPCTYPE* state2_cpu_copy = (CPPCTYPE*)malloc(sizeof(CPPCTYPE)*dim);
	get_quantum_state_host(state1_gpu, state1_cpu_copy, dim);
	copy_quantum_state_from_device_to_device(state2_gpu, state1_gpu, dim);
	get_quantum_state_host(state2_gpu, state2_cpu_copy, dim);

	for (UINT i = 0; i < dim; ++i) {
		ASSERT_NEAR(state1_cpu_copy[i].real(), state2_cpu_copy[i].real(), eps);
		ASSERT_NEAR(state1_cpu_copy[i].imag(), state2_cpu_copy[i].imag(), eps);
	}
	release_quantum_state_host(state1_gpu);
	release_quantum_state_host(state2_gpu);
}

TEST(StateTest, AddState) {
	const double eps = 1e-10;
	const UINT n = 10;
	QuantumStateGpu state1(n);
	QuantumStateGpu state2(n);
	state1.set_Haar_random_state();
	state2.set_Haar_random_state();

	const ITYPE dim = 1ULL << n;
	std::vector<std::complex<double>> state_vector1(dim);
	std::vector<std::complex<double>> state_vector2(dim);
	for (ITYPE i = 0; i < dim; ++i) {
		state_vector1[i] = state1.data_cpp()[i];
		state_vector2[i] = state2.data_cpp()[i];
	}

	state1.add_state(&state2);

	for (ITYPE i = 0; i < dim; ++i) {
		ASSERT_NEAR(state1.data_cpp()[i].real(), state_vector1[i].real() + state_vector2[i].real(), eps);
		ASSERT_NEAR(state1.data_cpp()[i].imag(), state_vector1[i].imag() + state_vector2[i].imag(), eps);
		ASSERT_NEAR(state2.data_cpp()[i].real(), state_vector2[i].real(), eps);
		ASSERT_NEAR(state2.data_cpp()[i].imag(), state_vector2[i].imag(), eps);
	}
}

TEST(StateTest, MultiplyCoef) {
	const double eps = 1e-10;
	const UINT n = 10;
	const std::complex<double> coef(0.5, 0.2);

	QuantumStateGpu state(n);
	state.set_Haar_random_state();

	const ITYPE dim = 1ULL << n;
	std::vector<std::complex<double>> state_vector(dim);
	for (ITYPE i = 0; i < dim; ++i) {
		state_vector[i] = state.data_cpp()[i] * coef;
	}
	state.multiply_coef(coef);

	for (ITYPE i = 0; i < dim; ++i) {
		ASSERT_NEAR(state.data_cpp()[i].real(), state_vector[i].real(), eps);
		ASSERT_NEAR(state.data_cpp()[i].imag(), state_vector[i].imag(), eps);
	}
}
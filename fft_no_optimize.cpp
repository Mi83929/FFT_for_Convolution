#include <chrono>
#include <complex>
using complex_t = std::complex<double>;

class Table {
public:
	inline complex_t get(int k, int log2N) {
		const double theta(2.0 * M_PI * k / log2N);
		return complex_t(cos(theta), -sin(theta));
	}
};

inline void Butterfly(complex_t* A, int s = 1) {
	const complex_t tmp(A[0]);
	A[0] = tmp + A[s];
	A[s] = tmp - A[s];
}

class FFT {
private:
	Table table;

public:
	FFT() : table() {}

	void _FFT_f(complex_t* A, int log2_BlockSize, int log2_N) {
		const int BlockSize(1 << log2_BlockSize), Num_of_Blocks(1 << (log2_N - log2_BlockSize)), stride(BlockSize >> 1);
		complex_t* A_(A);

		for (int n(0); n < Num_of_Blocks; ++n) {
			Butterfly(A_, stride);
			for (int i(1); i < stride; ++i) {
				Butterfly(A_ + i, stride);
				A_[i + stride] *= table.get(i, log2_BlockSize);
			}
			A_ += BlockSize;
		}
	}
	void _FFT_b(complex_t* A, int log2_BlockSize, int log2_N) {
		const int BlockSize(1 << log2_BlockSize), Num_of_Blocks(1 << (log2_N - log2_BlockSize)), stride(BlockSize >> 1);
		complex_t* A_(A);

		for (int n(0); n < Num_of_Blocks; ++n) {
			Butterfly(A_, stride);
			for (int i(1); i < stride; ++i) {
				A_[i + stride] *= conj(table.get(i, log2_BlockSize));
				Butterfly(A_ + i, stride);
			}
			A_ += BlockSize;
		}
	}

	void FFT_forward(complex_t* A, int log2_BlockSize, int log2_N) {
		if (log2_BlockSize == 0) return;

		_FFT_f(A, log2_BlockSize, log2_N);
		FFT_forward(A, log2_BlockSize - 1, log2_N);
	}
	void FFT_backward(complex_t* A, int log2_BlockSize, int log2_N) {
		if (log2_BlockSize == 0) return;

		FFT_backward(A, log2_BlockSize - 1, log2_N);
		_FFT_b(A, log2_BlockSize, log2_N);
	}
};

inline uint64_t get_time(void) {
	using namespace std::chrono;
	return (uint64_t)(static_cast<double>(duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count()));
}

int main() {
	int lmax(25), max(1 << lmax);
	complex_t* A = new complex_t[max];
	uint64_t start, end, fwd, bck, cnt;
	double fwd_, bck_;

	FFT fft;

	for (int i(0); i < max; ++i) A[i] = complex_t(rand(), rand());

	printf("|log2_N|FFT_forward/FFT_backward|NormalizedCost|\n|:-:|:-:|:-:|\n");

	for (int j(1); j <= lmax; ++j) {

		cnt = 0; fwd = 0; bck = 0;
		const double inv(1.0 / (1 << j));

		while (1) {
			++cnt;

			start = get_time();
			fft.FFT_forward(A, j, j);
			end = get_time();
			fwd += end - start;

			start = get_time();
			fft.FFT_backward(A, j, j);
			end = get_time();
			bck += end - start;

			for (int i(0); i < (1 << j); ++i) { A[i] /= ((double)(1 << j)); }

			if (fwd + bck > 10000000000ull) break;
		}
		fwd_ = (double)fwd / cnt; bck_ = (double)bck / cnt;
		
		printf("|%d|", j);
		
		if (fwd_ < 1e3) printf("%.3gns/", fwd_);
		else if (fwd_ < 1e6) printf("%.3gus/", 1e-3 * fwd_);
		else if (fwd_ < 1e9) printf("%.3gms/", 1e-6 * fwd_);
		else printf("%.3gs/", 1e-9 * fwd_);

		if (bck_ < 1e3) printf("%.3gns|", bck_);
		else if (bck_ < 1e6) printf("%.3gus|", 1e-3 * bck_);
		else if (bck_ < 1e9) printf("%.3gms|", 1e-6 * bck_);
		else printf("%.3gs|", 1e-9 * bck_);
		
		fwd_ /= (j << j); bck_ /= (j << j);
		printf("%.3g|", (fwd_ + bck_));

		printf("\n");
	}

	delete[] A;
	return 0;
}
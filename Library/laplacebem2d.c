/* ------------------------------------------------------------
 This is the file "laplacebem2d.c" of the H2Lib package.
 All rights reserved, Sven Christophersen 2011
 ------------------------------------------------------------ */

/**
 * @file laplacebem2d.c
 * @author Sven Christophersen
 * @date 2011
 */

/* C STD LIBRARY */
/* CORE 0 */
#include "basic.h"
#include "parameters.h"
/* CORE 1 */
/* CORE 2 */
/* CORE 3 */
/* SIMPLE */
/* PARTICLES */
/* BEM */
#include "laplacebem2d.h"

/* This constant comes from the fundamental solution of the Laplace-equation,
 * which is defined as: @f$ g(x,y) := - \frac{1}{2 \pi} \log \left( \left\lVert x - y \right\rVert \right)  @f$ .
 * Therefore the constant takes the value @f$- \frac{1}{2 \pi}@f$ .
 * */
#define KERNEL_CONST_BEM2D -0.159154943091895336
/* This constant comes from the fundamental solution of the Laplace-equation.
 * It is defined as @f$- \frac{1}{4 \pi}@f$ . Actually the fundamental solution
 * is defined as @f$ g(x,y) := - \frac{1}{2 \pi} \log \left( \left\lVert x - y \right\rVert \right)  @f$ .
 * But this is equal to
 * @f$ g(x,y) = - \frac{1}{2 \pi} \frac{1}{2} \log \left( \left\lVert x - y \right\rVert^2 \right)  @f$ ,
 * which lacks a squareroot in the computation of the vectornorm. The Performance
 * of our quadrature routines will benefit from this effect.
 * */
#define KERNEL_CONST_LOG_BEM2D -0.079577471545947668

/* ------------------------------------------------------------
 Nearfield entries for laplace-operator
 ------------------------------------------------------------ */

/*
 * @brief Computes nearfield entries of the slp operator with piecewise constant
 * basis functions.
 *
 * This function will compute nearfield entries of the slp operator using
 * piecewise constant basis functions @f$ \left( \varphi_i \right)_{i \in \mathcal I}
 * @f$ for the neumann data. In that case the
 * matrix entries are computed as:
 * @f[
 * N_{i,j} = \int_\Gamma \varphi_{\pi_r(i)} (\vec x) \, \int_\Gamma g(\vec x,
 * \, \vec y) \, \varphi_{\pi_c(j)}(\vec y) \, \mathrm d \vec y  \,\mathrm d \vec x
 * @f]
 * where @f$ \pi_r @f$ and @f$ \pi_c @f$ denote the index permutations described by the
 * arrays <tt>ridx</tt> and <tt>cidx</tt> respectively.
 *
 * @param ridx Index permutation of row indices. <tt>ridx[i] = </tt> @f$ \pi_r(i) @f$
 * The length of this array is determined by the rows of the matrix <tt>N</tt>.
 * This parameter can also be omitted with a <tt>NULL</tt> value. In that case
 * the permutation is assumed to be the identity.
 * @param cidx Index permutation of column indices. <tt>cidx[j] = </tt> @f$ \pi_c(j) @f$
 * The length of this array is determined by the columns of the matrix <tt>N</tt>.
 * This parameter can also be omitted with a <tt>NULL</tt> value. In that case
 * the permutation is assumed to be the identity.
 * @param bem @ref _bem2d "bem2d" object containing all necessary information
 * for computing the entries of <tt>N</tt> .
 * @param ntrans If <tt>ntrans</tt> equals true the entries will be stored in a
 * transposed way inside <tt>N</tt>.
 * @param N This matrix will contain the computed entries after calling this
 * function. The rows and columns of the matrix determine how many entries will
 * be computed.
 *
 * @attention Length of <tt>ridx</tt> must be at least <tt>N->rows</tt> if
 * <tt>ntrans == false</tt> or at least <tt>N->cols</tt> if <tt>ntrans == true</tt>
 * or <tt>ridx</tt> has to be <tt>NULL</tt>.
 * @attention Length of <tt>cidx</tt> must be at least <tt>N->cols</tt> if
 * <tt>ntrans == false</tt> or at least <tt>N->rows</tt> if <tt>ntrans == true</tt>
 * or <tt>ridx</tt> has to be <tt>NULL</tt>.
 */
static void
fill_slp_cc_laplacebem2d(const uint * ridx, const uint * cidx,
			 pcbem2d bem, bool ntrans, pamatrix N)
{
  const pccurve2d gr = bem->gr;
  const     real(*gr_x)[2] = (const real(*)[2]) gr->x;
  const     uint(*gr_e)[2] = (const uint(*)[2]) gr->e;
  const preal gr_g = (const preal) gr->g;
  field    *aa = N->a;
  uint      rows = N->rows;
  uint      cols = N->cols;
  longindex ld = N->ld;

  const real *A, *B, *C, *D;
  const uint *edge_t, *edge_s;
  real     *xq, *yq, *wq;
  uint      tp[2], sp[2];
  real      base, sum, tx, ty, dx, dy, factor, factor2;
  uint      c, q, nq, ss, tt;
  uint      t, s;

  if (ntrans == false) {
    for (s = 0; s < cols; ++s) {
      ss = (cidx == NULL ? s : cidx[s]);
      edge_s = gr_e[ss];
      factor = gr_g[ss] * KERNEL_CONST_LOG_BEM2D;
      for (t = 0; t < rows; ++t) {
	tt = (ridx == NULL ? t : ridx[t]);
	edge_t = gr_e[tt];
	factor2 = factor * gr_g[tt];

	c = select_quadrature_singquad1d(bem->sq, gr_e[tt], gr_e[ss], tp, sp,
					 &xq, &yq, &wq, &nq, &base);

	sum = 0.0;

	switch (c) {
	case 0:
	  A = gr_x[edge_t[0]];
	  B = gr_x[edge_t[1]];
	  C = gr_x[edge_s[0]];
	  D = gr_x[edge_s[1]];

	  for (q = 0; q < nq; ++q) {
	    tx = xq[q];
	    ty = yq[q];

	    dx = A[0] * tx + B[0] * (1.0 - tx)
	      - (C[0] * ty + D[0] * (1.0 - ty));
	    dy = A[1] * tx + B[1] * (1.0 - tx)
	      - (C[1] * ty + D[1] * (1.0 - ty));

	    sum += wq[q] * REAL_LOG(dx * dx + dy * dy);
	  }
	  break;
	case 1:
	  if (edge_t[0] == edge_s[0]) {
	    A = gr_x[edge_t[1]];
	    B = gr_x[edge_t[0]];
	    C = gr_x[edge_s[1]];
	  }
	  else if (edge_t[0] == edge_s[1]) {
	    A = gr_x[edge_t[1]];
	    B = gr_x[edge_t[0]];
	    C = gr_x[edge_s[0]];
	  }
	  else if (edge_t[1] == edge_s[0]) {
	    A = gr_x[edge_t[0]];
	    B = gr_x[edge_t[1]];
	    C = gr_x[edge_s[1]];
	  }
	  else if (edge_t[1] == edge_s[1]) {
	    A = gr_x[edge_t[0]];
	    B = gr_x[edge_t[1]];
	    C = gr_x[edge_s[0]];
	  }
	  else {
	    printf("ERROR!\n");
	    exit(0);
	  }

	  for (q = 0; q < nq; ++q) {
	    tx = xq[q];
	    ty = yq[q];

	    dx = A[0] * (-tx) + B[0] * (tx + ty) + C[0] * (-ty);
	    dy = A[1] * (-tx) + B[1] * (tx + ty) + C[1] * (-ty);

	    sum += wq[q] * REAL_LOG(dx * dx + dy * dy);
	  }

	  break;
	case 2:
	  sum += REAL_LOG(gr_g[tt]);

	  for (q = 0; q < nq; ++q) {
	    sum += wq[q] * REAL_LOG(REAL_ABS(xq[q] - yq[q]));
	  }
	  sum *= 2.0;
	  break;
	default:
	  break;
	}

	aa[t + s * ld] = (2.0 * base + sum) * factor2;
      }
    }
  }
  else {
    for (t = 0; t < cols; ++t) {
      tt = (ridx == NULL ? t : ridx[t]);
      edge_t = gr_e[tt];
      factor = gr_g[tt] * KERNEL_CONST_LOG_BEM2D;
      for (s = 0; s < rows; ++s) {
	ss = (cidx == NULL ? s : cidx[s]);
	edge_s = gr_e[ss];
	factor2 = factor * gr_g[ss];

	c = select_quadrature_singquad1d(bem->sq, gr_e[tt], gr_e[ss], tp, sp,
					 &xq, &yq, &wq, &nq, &base);

	sum = 0.0;

	switch (c) {
	case 0:
	  A = gr_x[edge_t[0]];
	  B = gr_x[edge_t[1]];
	  C = gr_x[edge_s[0]];
	  D = gr_x[edge_s[1]];

	  for (q = 0; q < nq; ++q) {
	    tx = xq[q];
	    ty = yq[q];

	    dx = A[0] * tx + B[0] * (1.0 - tx)
	      - (C[0] * ty + D[0] * (1.0 - ty));
	    dy = A[1] * tx + B[1] * (1.0 - tx)
	      - (C[1] * ty + D[1] * (1.0 - ty));

	    sum += wq[q] * REAL_LOG(dx * dx + dy * dy);
	  }
	  break;
	case 1:
	  if (edge_t[0] == edge_s[0]) {
	    A = gr_x[edge_t[1]];
	    B = gr_x[edge_t[0]];
	    C = gr_x[edge_s[1]];
	  }
	  else if (edge_t[0] == edge_s[1]) {
	    A = gr_x[edge_t[1]];
	    B = gr_x[edge_t[0]];
	    C = gr_x[edge_s[0]];
	  }
	  else if (edge_t[1] == edge_s[0]) {
	    A = gr_x[edge_t[0]];
	    B = gr_x[edge_t[1]];
	    C = gr_x[edge_s[1]];
	  }
	  else if (edge_t[1] == edge_s[1]) {
	    A = gr_x[edge_t[0]];
	    B = gr_x[edge_t[1]];
	    C = gr_x[edge_s[0]];
	  }
	  else {
	    printf("ERROR!\n");
	    exit(0);
	  }

	  for (q = 0; q < nq; ++q) {
	    tx = xq[q];
	    ty = yq[q];

	    dx = A[0] * (-tx) + B[0] * (tx + ty) + C[0] * (-ty);
	    dy = A[1] * (-tx) + B[1] * (tx + ty) + C[1] * (-ty);

	    sum += wq[q] * REAL_LOG(dx * dx + dy * dy);
	  }

	  break;
	case 2:
	  sum += REAL_LOG(gr_g[tt]);

	  for (q = 0; q < nq; ++q) {
	    sum += wq[q] * REAL_LOG(REAL_ABS(xq[q] - yq[q]));
	  }

	  sum *= 2.0;
	  break;
	default:
	  break;
	}

	aa[s + t * ld] = (2.0 * base + sum) * factor2;
      }
    }
  }
}

/*
 * @brief Computes nearfield entries of the dlp operator with piecewise constant
 * basis functions for either neumann and dirichlet data.
 *
 * This function will compute nearfield entries of the dlp operator using
 * piecewise constant basis functions @f$ \left( \varphi_i \right)_{i \in \mathcal I} @f$
 * for the neumann data and linear basis functions
 * @f$ \left( \psi_j \right)_{j \in \mathcal J} @f$ for the dirichlet data. In that case the
 * matrix entries are computed as:
 * @f[
 * N_{i,j} = \begin{cases} \begin{array}{ll}
 * \int_\Gamma \varphi_{\pi_r(i)} (\vec x) \, \int_\Gamma
 * \frac{\partial}{\partial \vec n_y} \, g(\vec x, \, \vec y) \,
 * \psi_{\pi_c(j)}(\vec y) \, \mathrm d \vec y  \,\mathrm d \vec x &: i \neq j\\
 * 0.5 \cdot \lvert \Gamma_i \rvert &: i = j
 * \end{array} \end{cases}
 * @f]
 * where @f$ \pi_r @f$ and @f$ \pi_c @f$ denote the index permutations described by the
 * arrays <tt>ridx</tt> and <tt>cidx</tt> respectively.
 *
 * @param ridx Index permutation of row indices. <tt>ridx[i] = </tt> @f$ \pi_r(i) @f$
 * The length of this array is determined by the rows of the matrix <tt>N</tt>.
 * This parameter can also be omitted with a <tt>NULL</tt> value. In that case
 * the permutation is assumed to be the identity.
 * @param cidx Index permutation of column indices. <tt>cidx[j] = </tt> @f$ \pi_c(j) @f$
 * The length of this array is determined by the columns of the matrix <tt>N</tt>.
 * This parameter can also be omitted with a <tt>NULL</tt> value. In that case
 * the permutation is assumed to be the identity.
 * @param bem @ref _bem2d "bem2d" object containing all necessary information
 * for computing the entries of <tt>N</tt> .
 * @param ntrans If <tt>ntrans</tt> equals true the entries will be stored in a
 * transposed way inside <tt>N</tt>.
 * @param N This matrix will contain the computed entries after calling this
 * function. The rows and columns of the matrix determine how many entries will
 * be computed.
 *
 * @attention Length of <tt>ridx</tt> must be at least <tt>N->rows</tt> if
 * <tt>ntrans == false</tt> or at least <tt>N->cols</tt> if <tt>ntrans == true</tt>
 * or <tt>ridx</tt> has to be <tt>NULL</tt>.
 * @attention Length of <tt>cidx</tt> must be at least <tt>N->cols</tt> if
 * <tt>ntrans == false</tt> or at least <tt>N->rows</tt> if <tt>ntrans == true</tt>
 * or <tt>ridx</tt> has to be <tt>NULL</tt>.
 */
static void
fill_dlp_cc_laplacebem2d(const uint * ridx, const uint * cidx,
			 pcbem2d bem, bool ntrans, pamatrix N)
{
  const pccurve2d gr = bem->gr;
  const     real(*gr_x)[2] = (const real(*)[2]) gr->x;
  const     uint(*gr_e)[2] = (const uint(*)[2]) gr->e;
  const     real(*gr_n)[2] = (const real(*)[2]) gr->n;
  const preal gr_g = (const preal) gr->g;
  field    *aa = N->a;
  uint      rows = N->rows;
  uint      cols = N->cols;
  longindex ld = N->ld;

  const real *A, *B, *C, *D, *n_s;
  const uint *edge_t, *edge_s;
  real     *xq, *yq, *wq;
  uint      tp[2], sp[2];
  real      norm2, sum, tx, ty, dx, dy, factor, factor2;
  uint      c, q, nq, ss, tt;
  uint      t, s;

  if (ntrans == false) {
    for (s = 0; s < cols; ++s) {
      ss = (cidx == NULL ? s : cidx[s]);
      edge_s = gr_e[ss];
      n_s = gr_n[ss];
      factor = gr_g[ss] * KERNEL_CONST_BEM2D;
      for (t = 0; t < rows; ++t) {
	tt = (ridx == NULL ? t : ridx[t]);
	edge_t = gr_e[tt];
	factor2 = factor * gr_g[tt];

	c = select_quadrature_singquad1d(bem->sq, gr_e[tt], gr_e[ss], tp, sp,
					 &xq, &yq, &wq, &nq, &sum);

	switch (c) {
	case 0:
	  A = gr_x[edge_t[0]];
	  B = gr_x[edge_t[1]];
	  C = gr_x[edge_s[0]];
	  D = gr_x[edge_s[1]];

	  for (q = 0; q < nq; ++q) {
	    tx = xq[q];
	    ty = yq[q];

	    dx = A[0] * tx + B[0] * (1.0 - tx)
	      - (C[0] * ty + D[0] * (1.0 - ty));
	    dy = A[1] * tx + B[1] * (1.0 - tx)
	      - (C[1] * ty + D[1] * (1.0 - ty));

	    norm2 = 1.0 / (dx * dx + dy * dy);

	    sum += wq[q] * (n_s[0] * dx + n_s[1] * dy) * norm2;
	  }
	  break;
	case 1:
	  if (edge_t[0] == edge_s[0]) {
	    A = gr_x[edge_t[1]];
	    B = gr_x[edge_t[0]];
	    C = gr_x[edge_s[1]];
	  }
	  else if (edge_t[0] == edge_s[1]) {
	    A = gr_x[edge_t[1]];
	    B = gr_x[edge_t[0]];
	    C = gr_x[edge_s[0]];
	  }
	  else if (edge_t[1] == edge_s[0]) {
	    A = gr_x[edge_t[0]];
	    B = gr_x[edge_t[1]];
	    C = gr_x[edge_s[1]];
	  }
	  else if (edge_t[1] == edge_s[1]) {
	    A = gr_x[edge_t[0]];
	    B = gr_x[edge_t[1]];
	    C = gr_x[edge_s[0]];
	  }
	  else {
	    printf("ERROR!\n");
	    exit(0);
	  }

	  for (q = 0; q < nq; ++q) {
	    tx = xq[q];
	    ty = yq[q];

	    dx = A[0] * (-tx) + B[0] * (tx + ty) + C[0] * (-ty);
	    dy = A[1] * (-tx) + B[1] * (tx + ty) + C[1] * (-ty);

	    sum += wq[q] * (n_s[0] * dx + n_s[1] * dy) / (dx * dx + dy * dy);
	  }

	  break;
	case 2:
	  factor2 = bem->alpha * gr_g[ss];
	  sum = 1.0;
	  break;
	default:
	  break;
	}

	aa[t + s * ld] = sum * factor2;

      }
    }
  }
  else {
    for (t = 0; t < cols; ++t) {
      tt = (ridx == NULL ? t : ridx[t]);
      edge_t = gr_e[tt];
      factor = gr_g[tt] * KERNEL_CONST_BEM2D;
      for (s = 0; s < rows; ++s) {
	ss = (cidx == NULL ? s : cidx[s]);
	edge_s = gr_e[ss];
	n_s = gr_n[ss];
	factor2 = factor * gr_g[ss];

	c = select_quadrature_singquad1d(bem->sq, gr_e[tt], gr_e[ss], tp, sp,
					 &xq, &yq, &wq, &nq, &sum);

	switch (c) {
	case 0:
	  A = gr_x[edge_t[0]];
	  B = gr_x[edge_t[1]];
	  C = gr_x[edge_s[0]];
	  D = gr_x[edge_s[1]];

	  for (q = 0; q < nq; ++q) {
	    tx = xq[q];
	    ty = yq[q];

	    dx = A[0] * tx + B[0] * (1.0 - tx)
	      - (C[0] * ty + D[0] * (1.0 - ty));
	    dy = A[1] * tx + B[1] * (1.0 - tx)
	      - (C[1] * ty + D[1] * (1.0 - ty));

	    sum += wq[q] * (n_s[0] * dx + n_s[1] * dy) / (dx * dx + dy * dy);
	  }
	  break;
	case 1:
	  if (edge_t[0] == edge_s[0]) {
	    A = gr_x[edge_t[1]];
	    B = gr_x[edge_t[0]];
	    C = gr_x[edge_s[1]];
	  }
	  else if (edge_t[0] == edge_s[1]) {
	    A = gr_x[edge_t[1]];
	    B = gr_x[edge_t[0]];
	    C = gr_x[edge_s[0]];
	  }
	  else if (edge_t[1] == edge_s[0]) {
	    A = gr_x[edge_t[0]];
	    B = gr_x[edge_t[1]];
	    C = gr_x[edge_s[1]];
	  }
	  else if (edge_t[1] == edge_s[1]) {
	    A = gr_x[edge_t[0]];
	    B = gr_x[edge_t[1]];
	    C = gr_x[edge_s[0]];
	  }
	  else {
	    printf("ERROR!\n");
	    exit(0);
	  }

	  for (q = 0; q < nq; ++q) {
	    tx = xq[q];
	    ty = yq[q];

	    dx = A[0] * (-tx) + B[0] * (tx + ty) + C[0] * (-ty);
	    dy = A[1] * (-tx) + B[1] * (tx + ty) + C[1] * (-ty);

	    sum += wq[q] * (n_s[0] * dx + n_s[1] * dy) / (dx * dx + dy * dy);
	  }

	  break;
	case 2:
	  factor2 = bem->alpha * gr_g[ss];
	  sum = 1.0;
	  break;
	default:
	  break;
	}

	aa[s + t * ld] = sum * factor2;
      }
    }
  }
}

/* ------------------------------------------------------------
 Integrals or evaluations of the kernel or its derivatives
 ------------------------------------------------------------ */

/*
 * @brief Evaluate the fundamental solution at certain points <tt>X</tt> and
 * <tt>Y</tt>.
 *
 * This function will fill the Matrix <tt>V</tt> by simply evaluating the
 * fundamental solution @f$ g @f$. Therefore the matrix entries will look like
 * @f[
 * V_{ij} = g( \texttt{X[i]}, \, \texttt{Y[j]})
 * @f]
 * with <tt>X[i]</tt> and <tt>Y[j]</tt> defining 2D-points.
 *
 * @param X An array of 2D-points whose length is defined by <tt>V->rows</tt>.
 * @param Y An array of 2D-points whose length is defined by <tt>V->cols</tt>.
 * @param V Matrix entries will be stored inside this matrix.
 *
 * @attention Length of <tt>X</tt> must be at least <tt>V->rows</tt>.
 * @attention Length of <tt>Y</tt> must be at least <tt>V->cols</tt>.
 */
static void
fill_kernel_laplacebem2d(const real(*X)[2], const real(*Y)[2], pamatrix V)
{
  uint      rows = V->rows;
  uint      cols = V->cols;
  longindex ld = V->ld;

  uint      i, j;
  real      dx, dy;

  for (j = 0; j < cols; ++j) {
    for (i = 0; i < rows; ++i) {
      dx = X[i][0] - Y[j][0];
      dy = X[i][1] - Y[j][1];

      V->a[i + j * ld] = KERNEL_CONST_LOG_BEM2D * REAL_LOG(dx * dx + dy * dy);
    }
  }
}

/*
 * @brief Evaluate the normal derivative of the fundamental solution in the
 * second argument at certain points <tt>X</tt> and <tt>Y</tt>.
 *
 * This function will fill the Matrix <tt>V</tt> by simply evaluating the
 * normal derivative of the fundamental solution in the
 * second argument @f$ \frac{\partial}{\partial \vec n_y} g @f$. Therefore
 * the matrix entries will look like
 * @f[
 * V_{ij} = \frac{\partial}{\partial \vec n_y} g( \texttt{X[i]}, \, \texttt{Y[j]})
 * @f]
 * with <tt>X[i]</tt> and <tt>Y[j]</tt> defining 2D-points.
 *
 * @param X An array of 2D-points whose length is defined by <tt>V->rows</tt>.
 * @param Y An array of 2D-points whose length is defined by <tt>V->cols</tt>.
 * @param NY An array of 2D normal vectors corresponding to <tt>Y</tt> whose
 * length is defined by <tt>V->cols</tt>.
 * @param V Matrix entries will be stored inside this matrix.
 *
 * @attention Length of <tt>X</tt> must be at least <tt>V->rows</tt>.
 * @attention Length of <tt>Y</tt> must be at least <tt>V->cols</tt>.
 * @attention Length of <tt>NY</tt> must be at least <tt>V->cols</tt>.
 */
static void
fill_dny_kernel_laplacebem2d(const real(*X)[2], const real(*Y)[2],
			     const real(*NY)[2], pamatrix V)
{
  uint      rows = V->rows;
  uint      cols = V->cols;
  longindex ld = V->ld;

  uint      i, j;
  real      norm2, dx, dy;

  for (j = 0; j < cols; ++j) {
    for (i = 0; i < rows; ++i) {
      dx = X[i][0] - Y[j][0];
      dy = X[i][1] - Y[j][1];

      norm2 = 1.0 / (dx * dx + dy * dy);

      V->a[i + j * ld] = KERNEL_CONST_BEM2D * (dx * NY[j][0] + dy * NY[j][1])
	* norm2;
    }
  }
}

/*
 * @brief Evaluate the normal derivatives of the fundamental solution in the
 * first and second argument at certain points <tt>X</tt> and <tt>Y</tt>.
 *
 * This function will fill the Matrix <tt>V</tt> by simply evaluating the
 * normal derivatives of the fundamental solution in the first and
 * second argument @f$ \frac{\partial^2}{\partial \vec n_x \, \partial \vec n_y}
 * g @f$. Therefore the matrix entries will look like
 * @f[
 * V_{ij} = \frac{\partial^2}{\partial \vec n_x \, \partial \vec n_y}
 * g( \texttt{X[i]}, \, \texttt{Y[j]})
 * @f]
 * with <tt>X[i]</tt> and <tt>Y[j]</tt> defining 2D-points.
 *
 * @param X An array of 2D-points whose length is defined by <tt>V->rows</tt>.
 * @param NX An array of 2D normal vectors corresponding to <tt>X</tt> whose
 * length is defined by <tt>V->rows</tt>.
 * @param Y An array of 2D-points whose length is defined by <tt>V->cols</tt>.
 * @param NY An array of 2D normal vectors corresponding to <tt>Y</tt> whose
 * length is defined by <tt>V->cols</tt>.
 * @param V Matrix entries will be stored inside this matrix.
 *
 * @attention Length of <tt>X</tt> must be at least <tt>V->rows</tt>.
 * @attention Length of <tt>NX</tt> must be at least <tt>V->rows</tt>.
 * @attention Length of <tt>Y</tt> must be at least <tt>V->cols</tt>.
 * @attention Length of <tt>NY</tt> must be at least <tt>V->cols</tt>.
 */
static void
fill_dnx_dny_kernel_laplacebem2d(const real(*X)[2],
				 const real(*NX)[2], const real(*Y)[2],
				 const real(*NY)[2], pamatrix V)
{
  uint      rows = V->rows;
  uint      cols = V->cols;
  longindex ld = V->ld;

  real      h[2];
  uint      i, j;
  real      norm2, dx, dy, dot;

  for (j = 0; j < cols; ++j) {
    for (i = 0; i < rows; ++i) {
      dx = X[i][0] - Y[j][0];
      dy = X[i][1] - Y[j][1];

      norm2 = 1.0 / (dx * dx + dy * dy);

      dot = dx * NY[j][0] + dy * NY[j][1];

      h[0] = NY[j][0] + 2.0 * dot * dx * norm2;
      h[1] = NY[j][1] + 2.0 * dot * dy * norm2;

      V->a[i + j * ld] = KERNEL_CONST_BEM2D
	* (NX[i][0] * h[0] + NX[i][1] * h[1]) * norm2;
    }
  }
}

/*
 * @brief Integrate the fundamental solution with piecewise constant basis functions.
 *
 * This function will fill the Matrix <tt>V</tt> by integrating the
 * fundamental solution @f$ g @f$ with piecewise constant basis functions
 * @f$ \left( \varphi_i \right)_{i \in \mathcal I} @f$.
 * Therefore the matrix entries will look like
 * @f[
 * V_{ij} = \int_\Gamma \, \varphi_{\pi(i)}(\vec x) \,
 * g( \vec x, \, \texttt{Z[j]}) \, \mathrm d \vec x
 * @f]
 * with <tt>Z[j]</tt> defining 2D-points.
 *
 * @param idx Index permutation of row indices. <tt>idx[i] = </tt> @f$ \pi(i) @f$
 * The length of this array is determined by the rows of the matrix <tt>V</tt>.
 * This parameter can also be omitted with a <tt>NULL</tt> value. In that case
 * the permutation is assumed to be the identity.
 * @param Z An array of 2D-points whose length is defined by <tt>V->cols</tt>.
 * @param bem @ref _bem2d "bem2d" object containing all necessary information
 * for computing the entries of <tt>V</tt> .
 * @param V Matrix entries will be stored inside this matrix.
 *
 * @attention Length of <tt>idx</tt> must be at least <tt>V->rows</tt>.
 * @attention Length of <tt>Z</tt> must be at least <tt>V->cols</tt>.
 */
static void
fill_kernel_c_laplacebem2d(const uint * idx, const real(*Z)[2],
			   pcbem2d bem, pamatrix V)
{
  pccurve2d gr = bem->gr;
  const     real(*gr_x)[2] = (const real(*)[2]) gr->x;
  const     uint(*gr_e)[2] = (const uint(*)[2]) gr->e;
  const preal gr_g = (const preal) gr->g;
  uint      rows = V->rows;
  uint      cols = V->cols;
  longindex ld = V->ld;

  uint      nq = bem->sq->n_single;
  real     *xx = bem->sq->x_single;
  real     *ww = bem->sq->w_single;

  const real *A, *B;
  uint      s, ss, i, q;
  real      gs_fac, sum, kernel, x, y, tx, Ax, Bx;

  /*
   *  integrate kernel function over first variable with constant basisfunctions
   */

  for (i = 0; i < cols; ++i) {
    for (s = 0; s < rows; ++s) {
      ss = (idx == NULL ? s : idx[s]);
      gs_fac = gr_g[ss] * KERNEL_CONST_LOG_BEM2D;
      A = gr_x[gr_e[ss][0]];
      B = gr_x[gr_e[ss][1]];

      sum = 0.0;

      for (q = 0; q < nq; ++q) {
	tx = xx[q];
	Ax = 1.0 - tx;
	Bx = tx;

	x = Z[i][0] - (A[0] * Ax + B[0] * Bx);
	y = Z[i][1] - (A[1] * Ax + B[1] * Bx);

	kernel = REAL_LOG(x * x + y * y);

	sum += ww[q] * kernel;
      }

      V->a[s + i * ld] = sum * gs_fac;
    }
  }
}

/*
 * @brief Integrate the normal derivative of the fundamental solution in the
 * first argument with piecewise constant basis functions.
 *
 * This function will fill the Matrix <tt>V</tt> by integrating the
 * normal derivative of the fundamental solution @f$ g @f$ in the first argument
 * with piecewise constant basis functions
 * @f$ \left( \varphi_i \right)_{i \in \mathcal I} @f$.
 * Therefore the matrix entries will look like
 * @f[
 * V_{ij} = \int_\Gamma \, \varphi_{\pi(i)}(\vec x) \,
 * \frac{\partial}{\partial \vec n_z} g( \texttt{Z[j]} , \, \vec x) \, \mathrm d \vec x
 * @f]
 * with <tt>Z[j]</tt> defining 2D-points.
 *
 * @param idx Index permutation of row indices. <tt>idx[i] = </tt> @f$ \pi(i) @f$
 * The length of this array is determined by the rows of the matrix <tt>V</tt>.
 * This parameter can also be omitted with a <tt>NULL</tt> value. In that case
 * the permutation is assumed to be the identity.
 * @param Z An array of 2D-points whose length is defined by <tt>V->cols</tt>.
 * @param N An array of 2D normal vectors corresponding to <tt>Z</tt> whose
 * length is defined by <tt>V->cols</tt>.
 * @param bem @ref _bem2d "bem2d" object containing all necessary information
 * for computing the entries of <tt>V</tt> .
 * @param V Matrix entries will be stored inside this matrix.
 *
 * @attention Length of <tt>idx</tt> must be at least <tt>V->rows</tt>.
 * @attention Length of <tt>Z</tt> must be at least <tt>V->cols</tt>.
 * @attention Length of <tt>N</tt> must be at least <tt>V->cols</tt>.
 */
static void
fill_dnz_kernel_c_laplacebem2d(const uint * idx, const real(*Z)[2],
			       const real(*N)[2], pcbem2d bem, pamatrix V)
{
  pccurve2d gr = bem->gr;
  const     real(*gr_x)[2] = (const real(*)[2]) gr->x;
  const     uint(*gr_e)[2] = (const uint(*)[2]) gr->e;
  const preal gr_g = (const preal) gr->g;
  uint      rows = V->rows;
  uint      cols = V->cols;
  longindex ld = V->ld;

  uint      nq = bem->sq->n_single;
  real     *xx = bem->sq->x_single;
  real     *ww = bem->sq->w_single;

  const real *A, *B;
  uint      t, tt, i, q;
  real      gt_fac, sum, kernel, dx, dy, tx, Ax, Bx;

  for (i = 0; i < cols; ++i) {
    for (t = 0; t < rows; ++t) {
      tt = (idx == NULL ? t : idx[t]);
      gt_fac = gr_g[tt] * KERNEL_CONST_BEM2D;
      A = gr_x[gr_e[tt][0]];
      B = gr_x[gr_e[tt][1]];

      sum = 0.0;

      for (q = 0; q < nq; ++q) {
	tx = xx[q];
	Ax = 1.0 - tx;
	Bx = tx;

	dx = Z[i][0] - (A[0] * Ax + B[0] * Bx);
	dy = Z[i][1] - (A[1] * Ax + B[1] * Bx);

	kernel = dx * dx + dy * dy;

	sum += ww[q] * (dx * N[i][0] + dy * N[i][1]) / kernel;

      }

      V->a[t + i * ld] = sum * gt_fac;
    }
  }
}

/*
 * @brief Integrate the normal derivative of the fundamental solution in the
 * second argument with piecewise constant basis functions.
 *
 * This function will fill the Matrix <tt>V</tt> by integrating the
 * normal derivative of the fundamental solution @f$ g @f$ in the second argument
 * with piecewise constant basis functions
 * @f$ \left( \varphi_i \right)_{i \in \mathcal I} @f$.
 * Therefore the matrix entries will look like
 * @f[
 * V_{ij} = \int_\Gamma \, \varphi_{\pi(i)}(\vec y) \,
 * \frac{\partial}{\partial \vec n_y} g( \texttt{Z[j]} , \, \vec y) \, \mathrm d \vec y
 * @f]
 * with <tt>Z[j]</tt> defining 2D-points.
 *
 * @param idx Index permutation of row indices. <tt>idx[i] = </tt> @f$ \pi(i) @f$
 * The length of this array is determined by the rows of the matrix <tt>V</tt>.
 * This parameter can also be omitted with a <tt>NULL</tt> value. In that case
 * the permutation is assumed to be the identity.
 * @param Z An array of 2D-points whose length is defined by <tt>V->cols</tt>.
 * @param bem @ref _bem2d "bem2d" object containing all necessary information
 * for computing the entries of <tt>V</tt> .
 * @param V Matrix entries will be stored inside this matrix.
 *
 * @attention Length of <tt>idx</tt> must be at least <tt>V->rows</tt>.
 * @attention Length of <tt>Z</tt> must be at least <tt>V->cols</tt>.
 */
static void
fill_dcol_kernel_col_c_laplacebem2d(const uint * idx,
				    const real(*Z)[2], pcbem2d bem,
				    pamatrix V)
{
  pccurve2d gr = bem->gr;
  const     real(*gr_x)[2] = (const real(*)[2]) gr->x;
  const     real(*gr_n)[2] = (const real(*)[2]) gr->n;
  const     uint(*gr_e)[2] = (const uint(*)[2]) gr->e;
  const preal gr_g = (const preal) gr->g;
  uint      rows = V->rows;
  uint      cols = V->cols;
  longindex ld = V->ld;

  uint      nq = bem->sq->n_single;
  real     *xx = bem->sq->x_single;
  real     *ww = bem->sq->w_single;

  const real *A, *B, *ns;
  uint      s, ss, i, q;
  real      gs_fac, sum, kernel, dx, dy, tx, Ax, Bx;

  /*
   *  integrate kernel function over first variable with constant basisfunctions
   */

  for (i = 0; i < cols; ++i) {
    for (s = 0; s < rows; ++s) {
      ss = (idx == NULL ? s : idx[s]);
      gs_fac = gr_g[ss] * KERNEL_CONST_BEM2D;
      ns = gr_n[ss];
      A = gr_x[gr_e[ss][0]];
      B = gr_x[gr_e[ss][1]];

      sum = 0.0;

      for (q = 0; q < nq; ++q) {
	tx = xx[q];

	Ax = 1.0 - tx;
	Bx = tx;

	dx = Z[i][0] - (A[0] * Ax + B[0] * Bx);
	dy = Z[i][1] - (A[1] * Ax + B[1] * Bx);

	kernel = dx * dx + dy * dy;

	sum += ww[q] * (dx * ns[0] + dy * ns[1]) / kernel;

      }

      V->a[s + i * ld] = sum * gs_fac;
    }
  }
}

/*
 * @brief Integrate the normal derivatives of the fundamental solution in the
 * first and second argument with piecewise constant basis functions.
 *
 * This function will fill the Matrix <tt>V</tt> by integrating the
 * normal derivatives of the fundamental solution @f$ g @f$ in the first and
 * second argument with piecewise constant basis functions
 * @f$ \left( \varphi_i \right)_{i \in \mathcal I} @f$.
 * Therefore the matrix entries will look like
 * @f[
 * V_{ij} = \int_\Gamma \, \varphi_{\pi(i)}(\vec y) \,
 * \frac{\partial^2}{\partial \vec n_z \, \partial \vec n_y}
 * g( \texttt{Z[j]} , \, \vec y) \, \mathrm d \vec y
 * @f]
 * with <tt>Z[j]</tt> defining 2D-points.
 *
 * @param idx Index permutation of row indices. <tt>idx[i] = </tt> @f$ \pi(i) @f$
 * The length of this array is determined by the rows of the matrix <tt>V</tt>.
 * This parameter can also be omitted with a <tt>NULL</tt> value. In that case
 * the permutation is assumed to be the identity.
 * @param Z An array of 2D-points whose length is defined by <tt>V->cols</tt>.
 * @param N An array of 2D normal vectors corresponding to <tt>Z</tt> whose
 * length is defined by <tt>V->cols</tt>.
 * @param bem @ref _bem2d "bem2d" object containing all necessary information
 * for computing the entries of <tt>V</tt> .
 * @param V Matrix entries will be stored inside this matrix.
 *
 * @attention Length of <tt>idx</tt> must be at least <tt>V->rows</tt>.
 * @attention Length of <tt>Z</tt> must be at least <tt>V->cols</tt>.
 * @attention Length of <tt>N</tt> must be at least <tt>V->cols</tt>.
 */
static void
fill_dnzdcol_kernel_c_laplacebem2d(const uint * idx,
				   const real(*Z)[2], const real(*N)[2],
				   pcbem2d bem, pamatrix V)
{
  pccurve2d gr = bem->gr;
  const     real(*gr_x)[2] = (const real(*)[2]) gr->x;
  const     real(*gr_n)[2] = (const real(*)[2]) gr->n;
  const     uint(*gr_e)[2] = (const uint(*)[2]) gr->e;
  const preal gr_g = (const preal) gr->g;
  uint      rows = V->rows;
  uint      cols = V->cols;
  longindex ld = V->ld;

  uint      nq = bem->sq->n_single;
  real     *xx = bem->sq->x_single;
  real     *ww = bem->sq->w_single;

  const real *A, *B, *ns;
  uint      s, ss, i, q;
  real      gs_fac, sum, kernel, norm2, tx, Ax, Bx, dotp1, dotp2, dotp3;
  real      dxy[2];

  for (i = 0; i < cols; ++i) {
    for (s = 0; s < rows; ++s) {
      ss = (idx == NULL ? s : idx[s]);
      gs_fac = gr_g[ss] * KERNEL_CONST_BEM2D;
      A = gr_x[gr_e[ss][0]];
      B = gr_x[gr_e[ss][1]];
      ns = gr_n[ss];

      sum = 0.0;

      for (q = 0; q < nq; ++q) {
	tx = xx[q];
	Ax = 1.0 - tx;
	Bx = tx;

	dxy[0] = Z[i][0] - (A[0] * Ax + B[0] * Bx);
	dxy[1] = Z[i][1] - (A[1] * Ax + B[1] * Bx);

	norm2 = 1.0 / (REAL_SQR(dxy[0]) + REAL_SQR(dxy[1]));

	dotp1 = N[i][0] * dxy[0] + N[i][1] * dxy[1];
	dotp2 = ns[0] * dxy[0] + ns[1] * dxy[1];
	dotp3 = ns[0] * N[i][0] + ns[1] * N[i][1];
	kernel = -2.0 * dotp1 * dotp2 * norm2 * norm2 + dotp3 * norm2;

	sum += ww[q] * kernel;

      }

      V->a[s + i * ld] = sum * gs_fac;
    }
  }
}

/* ------------------------------------------------------------
 Constructors and destructors
 ------------------------------------------------------------ */

pbem2d
new_slp_laplace_bem2d(pccurve2d gr, uint q, basisfunctionbem2d basis)
{
  pbem2d    bem;
  pkernelbem2d kernels;
  real     *x, *w;

  bem = new_bem2d(gr);
  kernels = bem->kernels;

  x = allocreal(q);
  w = allocreal(q);

  assemble_gauss(q, x, w);
  bem->sq = build_log_singquad1d(q, x, w);

  bem->basis_neumann = basis;

  kernels->fundamental = fill_kernel_laplacebem2d;
  kernels->dny_fundamental = fill_dny_kernel_laplacebem2d;
  kernels->dnx_dny_fundamental = fill_dnx_dny_kernel_laplacebem2d;

  if (basis == BASIS_CONSTANT_BEM2D) {
    bem->N_neumann = gr->edges;

    bem->nearfield = fill_slp_cc_laplacebem2d;

    kernels->lagrange_row = assemble_bem2d_lagrange_const_amatrix;
    kernels->lagrange_col = assemble_bem2d_lagrange_const_amatrix;

    kernels->fundamental_row = fill_kernel_c_laplacebem2d;
    kernels->fundamental_col = fill_kernel_c_laplacebem2d;
    kernels->dnz_fundamental_row = fill_dnz_kernel_c_laplacebem2d;
    kernels->dnz_fundamental_col = fill_dnz_kernel_c_laplacebem2d;

    kernels->kernel_row = fill_kernel_c_laplacebem2d;
    kernels->kernel_col = fill_kernel_c_laplacebem2d;
    kernels->dnz_kernel_row = fill_dnz_kernel_c_laplacebem2d;
    kernels->dnz_kernel_col = fill_dnz_kernel_c_laplacebem2d;
  }
  else {
    assert(basis == BASIS_LINEAR_BEM2D);
    bem->N_neumann = gr->vertices;

    bem->nearfield = NULL;

    kernels->lagrange_row = NULL;
    kernels->lagrange_col = NULL;

    kernels->fundamental_row = NULL;
    kernels->fundamental_col = NULL;
    kernels->dnz_fundamental_row = NULL;
    kernels->dnz_fundamental_col = NULL;

    kernels->kernel_row = NULL;
    kernels->kernel_col = NULL;
    kernels->dnz_kernel_row = NULL;
    kernels->dnz_kernel_col = NULL;
  }

  freemem(x);
  freemem(w);

  return bem;
}

pbem2d
new_dlp_laplace_bem2d(pccurve2d gr, uint q,
		      basisfunctionbem2d basis_neumann,
		      basisfunctionbem2d basis_dirichlet, field alpha)
{
  pbem2d    bem;
  pkernelbem2d kernels;
  real     *x, *w;

  bem = new_bem2d(gr);
  kernels = bem->kernels;

  x = allocreal(q);
  w = allocreal(q);

  assemble_gauss(q, x, w);
  bem->sq = build_pow_singquad1d(q, x, w, -1.0);

  bem->basis_neumann = basis_neumann;
  bem->basis_dirichlet = basis_dirichlet;
  bem->alpha = alpha;

  kernels->fundamental = fill_kernel_laplacebem2d;
  kernels->dny_fundamental = fill_dny_kernel_laplacebem2d;
  kernels->dnx_dny_fundamental = fill_dnx_dny_kernel_laplacebem2d;

  if (basis_neumann == BASIS_CONSTANT_BEM2D && basis_dirichlet
      == BASIS_CONSTANT_BEM2D) {
    bem->N_neumann = gr->edges;
    bem->N_dirichlet = gr->edges;

    bem->nearfield = fill_dlp_cc_laplacebem2d;

    kernels->lagrange_row = assemble_bem2d_lagrange_const_amatrix;
    kernels->lagrange_col = assemble_bem2d_dn_lagrange_const_amatrix;

    kernels->fundamental_row = fill_kernel_c_laplacebem2d;
    kernels->fundamental_col = fill_kernel_c_laplacebem2d;
    kernels->dnz_fundamental_row = fill_dnz_kernel_c_laplacebem2d;
    kernels->dnz_fundamental_col = fill_dnz_kernel_c_laplacebem2d;

    kernels->kernel_row = fill_kernel_c_laplacebem2d;
    kernels->kernel_col = fill_dcol_kernel_col_c_laplacebem2d;
    kernels->dnz_kernel_row = fill_dnz_kernel_c_laplacebem2d;
    kernels->dnz_kernel_col = fill_dnzdcol_kernel_c_laplacebem2d;
  }
  else if (basis_neumann == BASIS_CONSTANT_BEM2D
	   && basis_dirichlet == BASIS_LINEAR_BEM2D) {
    bem->N_neumann = gr->edges;
    bem->N_dirichlet = gr->vertices;

    bem->nearfield = NULL;

    kernels->lagrange_row = NULL;
    kernels->lagrange_col = NULL;

    kernels->fundamental_row = NULL;
    kernels->fundamental_col = NULL;
    kernels->dnz_fundamental_row = NULL;
    kernels->dnz_fundamental_col = NULL;

    kernels->kernel_row = NULL;
    kernels->kernel_col = NULL;
    kernels->dnz_kernel_row = NULL;
    kernels->dnz_kernel_col = NULL;
  }
  else if (basis_neumann == BASIS_LINEAR_BEM2D
	   && basis_dirichlet == BASIS_CONSTANT_BEM2D) {
    bem->N_neumann = gr->vertices;
    bem->N_dirichlet = gr->edges;

    bem->nearfield = NULL;

    kernels->lagrange_row = NULL;
    kernels->lagrange_col = NULL;

    kernels->fundamental_row = NULL;
    kernels->fundamental_col = NULL;
    kernels->dnz_fundamental_row = NULL;
    kernels->dnz_fundamental_col = NULL;

    kernels->kernel_row = NULL;
    kernels->kernel_col = NULL;
    kernels->dnz_kernel_row = NULL;
    kernels->dnz_kernel_col = NULL;
  }
  else {
    assert(basis_neumann == BASIS_LINEAR_BEM2D && basis_dirichlet
	   == BASIS_LINEAR_BEM2D);
    bem->N_neumann = gr->vertices;
    bem->N_dirichlet = gr->vertices;

    bem->nearfield = NULL;

    kernels->lagrange_row = NULL;
    kernels->lagrange_col = NULL;

    kernels->fundamental_row = NULL;
    kernels->fundamental_col = NULL;
    kernels->dnz_fundamental_row = NULL;
    kernels->dnz_fundamental_col = NULL;

    kernels->kernel_row = NULL;
    kernels->kernel_col = NULL;
    kernels->dnz_kernel_row = NULL;
    kernels->dnz_kernel_col = NULL;
  }

  freemem(x);
  freemem(w);

  return bem;
}

/* ------------------------------------------------------------
 Examples for Dirichlet- / Neumann-data to test linear system
 ------------------------------------------------------------ */

field
eval_dirichlet_linear_laplacebem2d(const real * x, const real * n)
{
  (void) n;

  return (field) x[0] - x[1];
}

field
eval_neumann_linear_laplacebem2d(const real * x, const real * n)
{
  (void) x;

  return (field) - (n[0] - n[1]);
}

field
eval_dirichlet_quadratic_laplacebem2d(const real * x, const real * n)
{
  (void) n;

  return (field) x[0] * x[0] - x[1] * x[1];
}

field
eval_neumann_quadratic_laplacebem2d(const real * x, const real * n)
{
  return (field) - 2.0 * (n[0] * x[0] - n[1] * x[1]);
}

matrixtype
build_interactive_laplacebem2d(pccurve2d gr, char op,
			       basisfunctionbem2d basis_neumann,
			       basisfunctionbem2d basis_dirichlet, uint q,
			       void **G, real * time, char *filename)
{
  pbem2d    bem = NULL;
  admissible admiss = NULL;
  quadpoints2d quadpoints = NULL;
  pstopwatch sw = NULL;
  pclustergeometry neumann_cg = NULL, dirichlet_cg = NULL;
  uint     *neumann_idx = NULL, *dirichlet_idx = NULL;
  pcluster  neumann = NULL, dirichlet = NULL;
  pclusterbasis neumann_cb = NULL, dirichlet_cb = NULL;
  pblock    tree = NULL;
  FILE     *file;
  real      eta = 0.0, eps = 0.0, delta = 0.0, accur_recomp =
    0.0, accur_coarsen = 0.0, accur_hiercomp = 0.0;
  field     alpha = 0.5;
  matrixtype huhh2 = HMATRIX, type = HMATRIX;
  uint      N = 0, M, clf = 0, method = 0, mi = 0, m = 0, l = 0;
  char      recomptech = 'n';
  bool      recomp = false, coarsen = false, hiercomp = false;

  *G = NULL;

  sw = new_stopwatch();

  huhh2 = HMATRIX;
  method = 1;
  eta = 1.0;
  mi = 2;
  m = 2;
  l = 1;
  delta = 0.5;
  eps = 1.0e-4;
  quadpoints = build_bem2d_rect_quadpoints;
  clf = 0;
  recomp = false;
  accur_recomp = 1.0e-4;
  recomptech = 'n';
  coarsen = false;
  accur_coarsen = 1.0e-4;
  hiercomp = false;
  accur_hiercomp = 1.0e-4;
  admiss = admissible_max_cluster;

  type = 0;

  if (op == 'd') {
    alpha = askforreal("alpha = ?", "h2lib_alpha_bem2d", alpha);
  }

  huhh2 = askforint("Type of initial matrix-approximation:\n"
		    "  [1] h-matrix,\n"
		    "  [2] h2-matrix?\n", "h2lib_approxtype_bem2d", huhh2);

  switch (huhh2) {
  case AMATRIX:
    printf("amatrix is not a valid approximation type!\n");
    abort();
    break;
  case HMATRIX:
    method = askforint("Method used for h-matrix-approximation:\n"
		       "  [ 1] Interpolation row-cluster\n"
		       "  [ 2] Interpolation col-cluster\n"
		       "  [ 3] Interpolation mixed\n"
		       "  [ 4] Green row-cluster\n"
		       "  [ 5] Green col-cluster\n"
		       "  [ 6] Green mixed\n"
		       "  [ 7] Greenhybrid row-cluster\n"
		       "  [ 8] Greenhybrid col-cluster\n"
		       "  [ 9] Greenhybrid mixed\n"
		       "  [10] ACA with full pivoting\n"
		       "  [11] ACA with partial pivoting\n"
		       "  [12] HCA\n", "h2lib_method_bem2d", method);

    switch (method) {
    case 1:			/* Interpolation row */
    case 2:			/* Interpolation col */
    case 3:			/* Interpolation mixed */
      mi = askforint("Interpolation order m?", "h2lib_mi_bem2d", mi);
      eta = 2.0;
      admiss = admissible_2_cluster;
      clf = 2 * mi * mi;
      clf = askforint("Minimal leafsize?", "h2lib_clf_bem2d", clf);
      if (filename != NULL) {
	if (!(file = fopen(filename, "r"))) {
	  file = fopen(filename, "a+");
	  fprintf(file, "#m\tclf\tsize\ttime\tabs error\trel error\n");
	}
	fclose(file);
	file = fopen(filename, "a+");
	fprintf(file, "%d\t%d\t", mi, clf);
	fclose(file);
      }
      break;
    case 4:			/* Green row */
    case 5:			/* Green col */
    case 6:			/* Green mixed */
      delta = 0.5;
      quadpoints =
	askforchar("[r]ect or [c]ircle parameterization?",
		   "h2lib_greenparam_bem2d", "rc", 'r')
	== 'r' ? build_bem2d_rect_quadpoints : build_bem2d_circle_quadpoints;
      admiss =
	(quadpoints == build_bem2d_rect_quadpoints) ?
	admissible_max_cluster : admissible_sphere_cluster;
      m = askforint("Quadrature order m?", "h2lib_m_bem2d", m);
      l = askforint("Number subdivisions l?", "h2lib_l_bem2d", l);
      delta = askforreal("delta = x * diam(t)?", "h2lib_delta_bem2d", delta);
      clf = 2 * (quadpoints == build_bem2d_rect_quadpoints ? 4 : 1) * m * l;
      clf = askforint("Minimal leafsize?", "h2lib_clf_bem2d", clf);
      if (filename != NULL) {
	if (!(file = fopen(filename, "r"))) {
	  file = fopen(filename, "a+");
	  fprintf(file,
		  "#m\tl\tdelta\tparam\tclf\tsize\ttime\tabs error\trel error\n");
	}
	fclose(file);
	file = fopen(filename, "a+");
	fprintf(file, "%d\t%d\t%.2f\t%c\t%d\t", m, l, delta,
		quadpoints == build_bem2d_rect_quadpoints ? 'r' : 'c', clf);
	fclose(file);
      }
      break;
    case 7:			/* Greenhybrid row */
    case 8:			/* Greenhybrid col */
    case 9:			/* Greenhybrid mixed */
      delta = 1.0;
      quadpoints =
	askforchar("[r]ect or [c]ircle parameterization?",
		   "h2lib_param_bem2d", "rc", 'r')
	== 'r' ? build_bem2d_rect_quadpoints : build_bem2d_circle_quadpoints;
      admiss =
	(quadpoints == build_bem2d_rect_quadpoints) ?
	admissible_max_cluster : admissible_sphere_cluster;
      m = askforint("Quadrature order m?", "h2lib_m_bem2d", m);
      l = askforint("Number subdivisions l?", "h2lib_l_bem2d", l);
      eps = askforreal("ACA accuracy?", "h2lib_eps_bem2d", eps);
      delta = askforreal("delta = x * diam(t)?", "h2lib_delta_bem2d", delta);
      clf = REAL_LOG(gr->edges);
      clf = askforint("Minimal leafsize?", "h2lib_clf_bem2d", clf);
      if (filename != NULL) {
	if (!(file = fopen(filename, "r"))) {
	  file = fopen(filename, "a+");
	  fprintf(file,
		  "#m\tl\tdelta\teps\tparam\tclf\tsize\ttime\tabs error\trel error\n");
	}
	fclose(file);
	file = fopen(filename, "a+");
	fprintf(file, "%d\t%d\t%.2f\t%.2e\t%c\t%d\t", m, l, delta, eps,
		quadpoints == build_bem2d_rect_quadpoints ? 'r' : 'c', clf);
	fclose(file);
      }
      break;
    case 10:			/* ACA full pivoting */
    case 11:			/* ACA partial pivoting */
      admiss = admissible_max_cluster;
      eps = askforreal("ACA accuracy?", "h2lib_eps_bem2d", eps);
      clf = REAL_LOG(gr->edges);
      clf = askforint("Minimal leafsize?", "h2lib_clf_bem2d", clf);
      if (filename != NULL) {
	if (!(file = fopen(filename, "r"))) {
	  file = fopen(filename, "a+");
	  fprintf(file, "#eps\tclf\tsize\ttime\tabs error\trel error\n");
	}
	fclose(file);
	file = fopen(filename, "a+");
	fprintf(file, "%.2e\t%d\t", eps, clf);
	fclose(file);
      }
      break;
    case 12:			/* HCA with interpolation */
      mi = askforint("Interpolation order m?", "h2lib_mi_bem2d", mi);
      eps = askforreal("ACA accuracy?", "h2lib_eps_bem2d", eps);
      eta = 2.0;
      admiss = admissible_2_cluster;
      clf = REAL_LOG(gr->edges);
      clf = askforint("Minimal leafsize?", "h2lib_clf_bem2d", clf);
      if (filename != NULL) {
	if (!(file = fopen(filename, "r"))) {
	  file = fopen(filename, "a+");
	  fprintf(file, "#m\teps\tclf\tsize\ttime\tabs error\trel error\n");
	}
	fclose(file);
	file = fopen(filename, "a+");
	fprintf(file, "%d\t%.2e\t%d\t", mi, eps, clf);
	fclose(file);
      }
      break;
    default:
      break;
    }
    break;
  case H2MATRIX:
    method = askforint("Method used for h2-matrix-approximation:\n"
		       "  [1] Interpolation\n"
		       "  [2] Greenhybrid\n"
		       "  [3] Greenhybrid ortho\n", "h2lib_method_bem2d",
		       method);

    switch (method) {
    case 1:			/* Interpolation */
      admiss = admissible_2_cluster;
      eta = 2.0;
      mi = askforint("Interpolation order m?", "h2lib_m_bem2d", mi);
      clf = 2 * mi * mi;
      clf = askforint("Minimal leafsize?", "h2lib_clf_bem2d", clf);
      if (filename != NULL) {
	if (!(file = fopen(filename, "r"))) {
	  file = fopen(filename, "a+");
	  fprintf(file, "#m\tclf\tsize\ttime\tabs error\trel error\n");
	}
	fclose(file);
	file = fopen(filename, "a+");
	fprintf(file, "%d\t%d\t", mi, clf);
	fclose(file);
      }
      break;
    case 2:			/* Greenhybrid */
      delta = 1.0;
      quadpoints =
	askforchar("[r]ect or [c]ircle parameterization?",
		   "h2lib_param_bem2d", "rc", 'r')
	== 'r' ? build_bem2d_rect_quadpoints : build_bem2d_circle_quadpoints;
      admiss =
	(quadpoints == build_bem2d_rect_quadpoints) ?
	admissible_max_cluster : admissible_sphere_cluster;
      m = askforint("Quadrature order m?", "h2lib_m_bem2d", m);
      l = askforint("Number subdivisions l?", "h2lib_l_bem2d", l);
      eps = askforreal("ACA accuracy?", "h2lib_eps_bem2d", eps);
      delta = askforreal("delta = x * diam(t)?", "h2lib_delta_bem2d", delta);
      clf = REAL_LOG(gr->edges);
      clf = askforint("Minimal leafsize?", "h2lib_clf_bem2d", clf);
      if (filename != NULL) {
	if (!(file = fopen(filename, "r"))) {
	  file = fopen(filename, "a+");
	  fprintf(file,
		  "#m\tl\tdelta\teps\tparam\tclf\tsize\ttime\tabs error\trel error\n");
	}
	fclose(file);
	file = fopen(filename, "a+");
	fprintf(file, "%d\t%d\t%.2f\t%.2e\t%c\t%d\t", m, l, delta, eps,
		quadpoints == build_bem2d_rect_quadpoints ? 'r' : 'c', clf);
	fclose(file);
      }
      break;
    case 3:			/* Greenhybrid ortho */
      delta = 1.0;
      quadpoints =
	askforchar("[r]ect or [c]ircle parameterization?",
		   "h2lib_greenparam_bem2d", "rc", 'r')
	== 'r' ? build_bem2d_rect_quadpoints : build_bem2d_circle_quadpoints;
      admiss =
	(quadpoints == build_bem2d_rect_quadpoints) ?
	admissible_max_cluster : admissible_sphere_cluster;
      m = askforint("Quadrature order m?", "h2lib_m_bem2d", m);
      l = askforint("Number subdivisions l?", "h2lib_l_bem2d", l);
      eps = askforreal("ACA accuracy?", "h2lib_eps_bem2d", eps);
      delta = askforreal("delta = x * diam(t)?", "h2lib_delta_bem2d", delta);
      clf = REAL_LOG(gr->edges);
      clf = askforint("Minimal leafsize?", "h2lib_clf_bem2d", clf);
      if (filename != NULL) {
	if (!(file = fopen(filename, "r"))) {
	  file = fopen(filename, "a+");
	  fprintf(file,
		  "#m\tl\tdelta\teps\tparam\tclf\tsize\ttime\tabs error\trel error\n");
	}
	fclose(file);
	file = fopen(filename, "a+");
	fprintf(file, "%d\t%d\t%.2f\t%.2e\t%c\t%d\t", m, l, delta, eps,
		quadpoints == build_bem2d_rect_quadpoints ? 'r' : 'c', clf);
	fclose(file);
      }
      break;
    default:
      break;
    }
    break;
  default:
    printf("ERROR: unknown Matrix-type!\n");
    abort();
    break;
  }

  if (askforchar("Recompression?", "h2lib_recomp_bem2d", "yn",
		 recomp ? 'y' : 'n')
      == 'y') {
    accur_recomp = askforreal("Recompression accuracy?",
			      "h2lib_accur_recomp_bem2d", accur_recomp);
    recomp = true;
  }
  else {
    recomp = false;
  }

  if (huhh2 == HMATRIX) {
    recomptech = askforchar("Further recompression techniques ?\n"
			    "  [c]oarsen,\n"
			    "  [h]ierarchical compression,\n"
			    "  [n]one?\n", "h2lib_recomptech_bem2d", "chn",
			    recomptech);
    if (recomptech == 'c') {
      coarsen = true;
      accur_coarsen = askforreal("Coarsen accuracy?",
				 "h2lib_accur_coarsen_bem2d", accur_coarsen);
      hiercomp = false;
      accur_hiercomp = 0.0;
    }
    else if (recomptech == 'h') {
      coarsen = false;
      accur_coarsen = 0.0;
      hiercomp = true;
      accur_hiercomp = askforreal("Hierarchical compression accuracy?",
				  "h2lib_accur_hiercomp_bem2d",
				  accur_hiercomp);
    }
    else {
      coarsen = false;
      accur_coarsen = 0.0;
      hiercomp = false;
      accur_hiercomp = 0.0;
    }
  }

  if (op == 's') {
    bem = new_slp_laplace_bem2d(gr, q, basis_neumann);
  }
  else {
    assert(op == 'd');
    bem = new_dlp_laplace_bem2d(gr, q, basis_neumann, basis_dirichlet, alpha);
  }

  N = bem->N_neumann;
  M = bem->N_dirichlet;

  start_stopwatch(sw);

  neumann_cg = build_bem2d_clustergeometry(bem, &neumann_idx,
					   bem->basis_neumann);
  neumann = build_adaptive_cluster(neumann_cg, N, neumann_idx, clf);
  dirichlet = neumann;
  del_clustergeometry(neumann_cg);

  if (op == 'd') {
    dirichlet_cg = build_bem2d_clustergeometry(bem, &dirichlet_idx,
					       bem->basis_dirichlet);
    dirichlet = build_adaptive_cluster(dirichlet_cg, M, dirichlet_idx, clf);
    del_clustergeometry(dirichlet_cg);
  }

  switch (huhh2) {
  case AMATRIX:
    printf("amatrix is not a valid approximation type!\n");
    abort();
    break;
  case HMATRIX:
    tree = build_nonstrict_block(neumann, dirichlet, (void *) &eta, admiss);

    /* h-matrix approximations */
    switch (method) {
    case 1:			/* Interpolation row */
      setup_hmatrix_aprx_inter_row_bem2d(bem, neumann, dirichlet, tree, mi);
      setup_hmatrix_recomp_bem2d(bem, recomp, accur_recomp, coarsen,
				 accur_coarsen);
      break;
    case 2:			/* Interpolation col */
      setup_hmatrix_aprx_inter_col_bem2d(bem, neumann, dirichlet, tree, mi);
      setup_hmatrix_recomp_bem2d(bem, recomp, accur_recomp, coarsen,
				 accur_coarsen);
      break;
    case 3:			/* Interpolation mixed */
      setup_hmatrix_aprx_inter_mixed_bem2d(bem, neumann, dirichlet, tree, mi);
      setup_hmatrix_recomp_bem2d(bem, recomp, accur_recomp, coarsen,
				 accur_coarsen);
      break;
    case 4:			/* Green row */
      setup_hmatrix_aprx_green_row_bem2d(bem, neumann, dirichlet, tree, m, l,
					 delta, quadpoints);
      setup_hmatrix_recomp_bem2d(bem, recomp, accur_recomp, coarsen,
				 accur_coarsen);
      break;
    case 5:			/* Green col */
      setup_hmatrix_aprx_green_col_bem2d(bem, neumann, dirichlet, tree, m, l,
					 delta, quadpoints);
      setup_hmatrix_recomp_bem2d(bem, recomp, accur_recomp, coarsen,
				 accur_coarsen);
      break;
    case 6:			/* Green mixed */
      setup_hmatrix_aprx_green_mixed_bem2d(bem, neumann, dirichlet, tree, m,
					   l, delta, quadpoints);
      setup_hmatrix_recomp_bem2d(bem, recomp, accur_recomp, coarsen,
				 accur_coarsen);
      break;
    case 7:			/* Greenhybrid row */
      setup_hmatrix_aprx_greenhybrid_row_bem2d(bem, neumann, dirichlet, tree,
					       m, l, delta, eps, quadpoints);
      setup_hmatrix_recomp_bem2d(bem, recomp, accur_recomp, coarsen,
				 accur_coarsen);
      break;
    case 8:			/* Greenhybrid col */
      setup_hmatrix_aprx_greenhybrid_col_bem2d(bem, neumann, dirichlet, tree,
					       m, l, delta, eps, quadpoints);
      setup_hmatrix_recomp_bem2d(bem, recomp, accur_recomp, coarsen,
				 accur_coarsen);
      break;
    case 9:			/* Greenhybrid mixed */
      setup_hmatrix_aprx_greenhybrid_mixed_bem2d(bem, neumann, dirichlet,
						 tree, m, l, delta, eps,
						 quadpoints);
      setup_hmatrix_recomp_bem2d(bem, recomp, accur_recomp, coarsen,
				 accur_coarsen);
      break;
    case 10:			/* ACA full pivoting */
      setup_hmatrix_aprx_aca_bem2d(bem, neumann, dirichlet, tree, eps);
      setup_hmatrix_recomp_bem2d(bem, recomp, accur_recomp, coarsen,
				 accur_coarsen);
      break;
    case 11:			/* ACA partial pivoting */
      setup_hmatrix_aprx_paca_bem2d(bem, neumann, dirichlet, tree, eps);
      setup_hmatrix_recomp_bem2d(bem, recomp, accur_recomp, coarsen,
				 accur_coarsen);
      break;
    case 12:			/* HCA with interpolation */
      setup_hmatrix_aprx_hca_bem2d(bem, neumann, dirichlet, tree, mi, eps);
      setup_hmatrix_recomp_bem2d(bem, recomp, accur_recomp, coarsen,
				 accur_coarsen);
      break;
    default:
      break;
    }

    if (coarsen) {
      *G = (void *) build_from_block_hmatrix(tree, 0);
      type = HMATRIX;
      assemblecoarsen_bem2d_hmatrix(bem, tree, (phmatrix) *G);
    }
    else if (hiercomp) {
      del_block(tree);
      tree = build_strict_block(neumann, dirichlet, (void *) &eta, admiss);
      neumann_cb = build_from_cluster_clusterbasis(neumann);
      dirichlet_cb = neumann_cb;
      if (op == 'd') {
	dirichlet_cb = build_from_cluster_clusterbasis(dirichlet);
      }
      *G = (void *) build_from_block_h2matrix(tree, neumann_cb, dirichlet_cb);
      type = H2MATRIX;
      setup_h2matrix_recomp_bem2d(bem, hiercomp, accur_hiercomp);
      assemblehiercomp_bem2d_h2matrix(bem, tree, (ph2matrix) *G);
    }
    else {
      *G = (void *) build_from_block_hmatrix(tree, 0);
      type = HMATRIX;
      assemble_bem2d_hmatrix(bem, tree, (phmatrix) *G);
    }

    break;
  case H2MATRIX:
    type = H2MATRIX;

    neumann_cb = build_from_cluster_clusterbasis(neumann);
    dirichlet_cb = build_from_cluster_clusterbasis(dirichlet);

    tree = build_strict_block(neumann, dirichlet, (void *) &eta, admiss);
    *G = (void *) build_from_block_h2matrix(tree, neumann_cb, dirichlet_cb);

    /* H2-matrix approximations */
    switch (method) {
    case 1:			/* Interpolation H2 */
      setup_h2matrix_aprx_inter_bem2d(bem, neumann_cb, dirichlet_cb, tree,
				      mi);
      break;
    case 2:			/* Greenhybrid H2 */
      setup_h2matrix_aprx_greenhybrid_bem2d(bem, neumann_cb, dirichlet_cb,
					    tree, m, l, delta, eps,
					    quadpoints);
      break;
    case 3:			/* Greenhybrid ortho H2 */
      setup_h2matrix_aprx_greenhybrid_ortho_bem2d(bem, neumann_cb,
						  dirichlet_cb, tree, m, l,
						  delta, eps, quadpoints);
      break;
    default:
      break;
    }

    assemble_bem2d_h2matrix_row_clusterbasis(bem, neumann_cb);
    assemble_bem2d_h2matrix_col_clusterbasis(bem, dirichlet_cb);

    assemble_bem2d_h2matrix(bem, tree, (ph2matrix) *G);

    if (recomp == true) {
      recompress_inplace_h2matrix((ph2matrix) *G, 0, accur_recomp);
    }
    break;
  }

  *time += stop_stopwatch(sw);

  del_stopwatch(sw);
  del_bem2d(bem);
  del_block(tree);

  return type;
}

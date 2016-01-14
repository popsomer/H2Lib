#include "basic.h"
#include "krylov.h"
#include "helmholtzbem3d.h"

#define IS_IN_RANGE(a, b, c) (((a) <= (b)) && ((b) <= (c)))

struct _eval_A {
  matrixtype Vtype;
  const void     *V;
  matrixtype KMtype;
  const void     *KM;
  field     eta;
};

/* eval Brakhage-Werner system matrix */
void
addeval_A(field alpha, void *matrix, pcavector x, pavector y)
{
  struct _eval_A *eval = (struct _eval_A *) matrix;

  field     beta;

  beta = -I * alpha * eval->eta;

  switch (eval->KMtype) {
  case AMATRIX:
    addeval_amatrix_avector(alpha, (pamatrix) eval->KM, x, y);
    break;
  case HMATRIX:
    addeval_hmatrix_avector(alpha, (phmatrix) eval->KM, x, y);
    break;
  case H2MATRIX:
    addeval_h2matrix_avector(alpha, (ph2matrix) eval->KM, x, y);
    break;
  default:
    printf("ERROR: unknown matrix type!\n");
    abort();
    break;
  }

  switch (eval->Vtype) {
  case AMATRIX:
    addeval_amatrix_avector(beta, (pamatrix) eval->V, x, y);
    break;
  case HMATRIX:
    addeval_hmatrix_avector(beta, (phmatrix) eval->V, x, y);
    break;
  case H2MATRIX:
    addeval_h2matrix_avector(beta, (ph2matrix) eval->V, x, y);
    break;
  default:
    printf("ERROR: unknown matrix type!\n");
    abort();
    break;
  }
}

/* Simple convenience wrapper for GMRES solver */
static void
solve_gmres_bem3d(matrixtype type, void *A, pavector b, pavector x,
		  real accuracy, uint steps)
{
  addeval_t addevalA;
  pavector  rhat, q, tau;
  pamatrix  qr;
  uint      i, j, n, kk, kmax;
  real      norm;

  addevalA = (addeval_t) addeval_A;

  kmax = 500;

  n = b->dim;
  assert(x->dim == n);

  qr = new_zero_amatrix(n, kmax);
  rhat = new_avector(n);
  q = new_avector(n);
  tau = new_avector(kmax);
  clear_avector(x);

  init_gmres(addevalA, A, b, x, rhat, q, &kk, qr, tau);

  for (i = 0; i < steps; i += kmax) {
    for (j = 0; j < kmax && i + j < steps; ++j) {
      step_gmres(addevalA, A, b, x, rhat, q, &kk, qr, tau);
      norm = residualnorm_gmres(rhat, kk);
      if(false) {
	printf("  Residual: %.5e\t Iterations: %u\r", norm, j + i);
	fflush(stdout);
      }
      if (norm <= accuracy) {
	finish_gmres(addevalA, A, b, x, rhat, q, &kk, qr, tau);
	break;
      }
    }
    if (norm <= accuracy) {
      break;
    }
    else {
      finish_gmres(addevalA, A, b, x, rhat, q, &kk, qr, tau);
    }
  }

  printf("\n");
  del_avector(rhat);
  del_avector(q);
  del_avector(tau);
  del_amatrix(qr);

}

field
eval_brakhage_werner_c(pcbem3d bem, pcavector w, field eta, real * x)
{
  pcsurface3d gr = bem->gr;
  const     real(*gr_x)[3] = (const real(*)[3]) gr->x;
  const     uint(*gr_t)[3] = (const uint(*)[3]) gr->t;
  const     real(*gr_n)[3] = (const real(*)[3]) gr->n;
  const preal gr_g = (const preal) gr->g;
  const uint rows = gr->triangles;

  uint      nq = bem->sq->n_single;
  real     *xx = bem->sq->x_single;
  real     *yy = bem->sq->y_single;
  real     *ww = bem->sq->w_single + 3 * nq;

  const real *A, *B, *C, *ns;
  uint      s, ss, q;
  real      gs_fac, dx, dy, dz, tx, sx, Ax, Bx, Cx, norm, rnorm, norm2;
  field     k, sum;

  field     res;

  k = bem->k;

  res = 0.0;

  for (s = 0; s < rows; ++s) {
    ss = s;
    gs_fac = gr_g[ss] * bem->kernel_const;
    ns = gr_n[ss];
    A = gr_x[gr_t[ss][0]];
    B = gr_x[gr_t[ss][1]];
    C = gr_x[gr_t[ss][2]];

    sum = 0.0;

    for (q = 0; q < nq; ++q) {
      tx = xx[q];
      sx = yy[q];
      Ax = 1.0 - tx;
      Bx = tx - sx;
      Cx = sx;

      dx = x[0] - (A[0] * Ax + B[0] * Bx + C[0] * Cx);
      dy = x[1] - (A[1] * Ax + B[1] * Bx + C[1] * Cx);
      dz = x[2] - (A[2] * Ax + B[2] * Bx + C[2] * Cx);

      norm2 = dx * dx + dy * dy + dz * dz;
      rnorm = REAL_RSQRT(norm2);
      norm = norm2 * rnorm;
      rnorm *= rnorm * rnorm;

      sum += ww[q]
	* cexp(I * k * norm)
	* rnorm
	* ((1.0 - I * k * norm) * (dx * ns[0] + dy * ns[1] + dz * ns[2]) - I
	   * eta * norm2);
    }

    res += sum * gs_fac * w->v[ss];
  }

  return res;
}

field
eval_brakhage_werner_l(pcbem3d bem, pcavector w, field eta, real * x)
{
  pcsurface3d gr = bem->gr;
  const     real(*gr_x)[3] = (const real(*)[3]) gr->x;
  const     uint(*gr_t)[3] = (const uint(*)[3]) gr->t;
  const     real(*gr_n)[3] = (const real(*)[3]) gr->n;
  const preal gr_g = (const preal) gr->g;
  const uint triangles = gr->triangles;
  const uint vertices = gr->vertices;
  real      k = bem->k;

  uint      nq = bem->sq->n_single;
  real     *xx = bem->sq->x_single;
  real     *yy = bem->sq->y_single;
  real     *ww = bem->sq->w_single;
  real      base = bem->sq->base_single;

  pavector  v;
  const real *A, *B, *C, *N;
  uint      s, i, ii, q;
  real      gs_fac, tx, sx, Ax, Bx, Cx, norm, rnorm, norm2, dx[3];
  field     sum;
  field    *quad;

  field     res;

  assert(gr->vertices == w->dim);

  quad = allocfield(nq);
  v = new_zero_avector(vertices);

  for (s = 0; s < triangles; s++) {
    gs_fac = gr_g[s] * bem->kernel_const;
    A = gr_x[gr_t[s][0]];
    B = gr_x[gr_t[s][1]];
    C = gr_x[gr_t[s][2]];
    N = gr_n[s];

    for (q = 0; q < nq; ++q) {
      tx = xx[q];
      sx = yy[q];
      Ax = 1.0 - tx;
      Bx = tx - sx;
      Cx = sx;

      dx[0] = x[0] - (A[0] * Ax + B[0] * Bx + C[0] * Cx);
      dx[1] = x[1] - (A[1] * Ax + B[1] * Bx + C[1] * Cx);
      dx[2] = x[2] - (A[2] * Ax + B[2] * Bx + C[2] * Cx);

      norm2 = REAL_NORMSQR3(dx[0], dx[1], dx[2]);
      rnorm = REAL_RSQRT(norm2);
      norm = norm2 * rnorm;
      rnorm *= rnorm * rnorm;

      quad[q] =
	cexp(I * k * norm) * rnorm
	* ((1.0 - I * k * norm) *
	   (dx[0] * N[0] + dx[1] * N[1] + dx[2] * N[2]) - I * eta * norm2);
    }

    ww = bem->sq->w_single;

    for (i = 0; i < 3; ++i) {
      ii = gr_t[s][i];
      sum = base;

      for (q = 0; q < nq; ++q) {
	sum += ww[q] * quad[q];
      }

      assert(ii < vertices);
      v->v[ii] += sum * gs_fac;

      ww += nq;
    }
  }

  res = 0.0;
  for (i = 0; i < vertices; ++i) {
    res += w->v[i] * v->v[i];
  }

  del_avector(v);
  freemem(quad);

  return res;
}

real
max_rel_outer_error(pcbem3d bem, helmholtz_data * hdata, pcavector x,
		    boundary_func3d rhs, basisfunctionbem3d basis)
{
  uint      nx, nz, npoints;
  real(*xdata)[3];
  field    *ydata;
  uint      i, j;
  real      error, maxerror;
  real      eta_bw = bem->k;

  nx = 20;
  nz = 20;
  npoints = nx * nz;

  xdata = (real(*)[3]) allocreal(3 * npoints);
  npoints = 0;
  for (j = 0; j < nz; ++j) {
    for (i = 0; i < nx; ++i) {
      xdata[npoints][0] = -10.0 + (20.0 / (nx - 1)) * i;
      xdata[npoints][1] = 0.0;
      xdata[npoints][2] = -10.0 + (20.0 / (nz - 1)) * j;
      if (REAL_SQR(xdata[npoints][0]) + REAL_SQR(xdata[npoints][2]) > 1) {
	npoints++;
      }
    }
  }

  ydata = allocfield(npoints);

  if (basis == BASIS_CONSTANT_BEM3D) {
    for (j = 0; j < npoints; ++j) {
      ydata[j] = eval_brakhage_werner_c(bem, x, eta_bw, xdata[j]);
    }
  }
  else {
    assert(basis == BASIS_LINEAR_BEM3D);
    for (j = 0; j < npoints; ++j) {
      ydata[j] = eval_brakhage_werner_l(bem, x, eta_bw, xdata[j]);
    }
  }

  j = 0;
  maxerror =
    ABS(ydata[j] -
	rhs(xdata[j], NULL, hdata)) / ABS(rhs(xdata[j], NULL, hdata));
  for (j = 1; j < npoints; ++j) {
    error =
      ABS(ydata[j] -
	  rhs(xdata[j], NULL, hdata)) / ABS(rhs(xdata[j], NULL, hdata));
    maxerror = error > maxerror ? error : maxerror;
  }

  freemem(ydata);
  freemem(xdata);

  return maxerror;
}



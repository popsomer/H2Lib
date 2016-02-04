#include "basic.h"
#include "krylov.h"
#include "helmholtzbem3d.h"
#include "validation.c"

#define IS_IN_RANGE(a, b, c) (((a) <= (b)) && ((b) <= (c)))

const bool* printed;// = &false;
//bool* const printed; // Pointer is constant, data is not
const real a = 2.2;
const real b = 2.8;

static inline field
slp_kernel_helmholtzbem3d_compr(const real * x, const real * y, const real * nx, const real * ny, void *data)
{
  pcbem3d   bem = (pcbem3d) data;
  real      k = bem->k;
  real      dist[3];
  real      norm, norm2, rnorm, norTest, norRay;
  field     res;

  (void) nx;
  (void) ny;

  dist[0] = x[0] - y[0];
  dist[1] = x[1] - y[1];
  dist[2] = x[2] - y[2];
  norm2 = REAL_NORMSQR3(dist[0], dist[1], dist[2]);
  rnorm = REAL_RSQRT(norm2);

  norm = k * norm2 * rnorm;
  res = (cos(norm) + I * sin(norm)) * rnorm;

  norTest = REAL_NORMSQR3(x[0],x[1],x[2]);
  norRay =  REAL_NORMSQR3(y[0],y[1],y[2]);
  if((abs(norTest-1) > 1e-5) || (abs(norRay-1) > 1e-5)) {
     printf("error, norm2 is %g and norRay is %g.\n", norTest, norRay);
     exit(1);
  }
  
  norRay = sqrt(pow(x[1]-y[1],2)+pow(x[2]-y[2],2));
  if(norRay > b) {
     res = 0.0;
  }
  else if(norRay > a) {
     res *= exp(2*exp(-(b-a)/(norRay-a) )/((norRay-a)/(b-a) -1));
  }
  if(!printed) {
//  if(x[0] > 0.8) {
     printf("hello %g \n", x[0]);
     printed = ((bool*) true);
//     *printed = true;
     printf("hello %g \n", x[0]);
//     printed = (bool*) true;
  }
  return res;
}

int
main(int argc, char **argv)
{
  pmacrosurface3d mg;
  psurface3d gr;
  uint      n, q, clf, j, i, sys;
  real      eta; //, elmt, sum;
  field     kvec[3], elmt, sum, norb, norerr;
  
  struct _eval_A eval;
  struct _eval_A evalf;
  struct _eval_A evalfC;
  pavector  x, b, xf, xfc, tst;
  helmholtz_data hdata;
  real      errorV, errorKM, error_solve, eps_solve, errSol;
  uint      steps;
  boundary_func3d rhs = (boundary_func3d) rhs_dirichlet_point_helmholtzbem3d;

  pbem3d    bem_slp, bem_slpCompr, bem_dlp;
  pcluster  rootn, rootd;
  pblock    brootV, brootKM;
  pamatrix  VfullCompr, Vfull, KMfull;
  phmatrix  V, KM;
  uint      nn, nd;
  uint      m;

  basisfunctionbem3d basis_neumann;
  basisfunctionbem3d basis_dirichlet;
  bool exterior;

  init_h2lib(&argc, &argv);
  n = 512;
  kvec[0] = 2.0, kvec[1] = 0.0, kvec[2] = 0.0;
  q = 2;
  eta = 1.0;

  mg = new_sphere_macrosurface3d();
  gr = build_from_macrosurface3d_surface3d(mg, REAL_SQRT(n * 0.125));
  n = gr->triangles;

  write_surface3d(gr, "../H2Lib_trunk/sphere.tri");
  printf("Testing unit sphere with %d triangles and %d vertices\n", gr->triangles, gr->vertices);
  clf = 16;


  basis_neumann = BASIS_LINEAR_BEM3D;
  basis_dirichlet = BASIS_LINEAR_BEM3D;
  exterior = true;

  nn = basis_neumann == BASIS_LINEAR_BEM3D ? gr->vertices : gr->triangles;
  nd = basis_dirichlet == BASIS_LINEAR_BEM3D ? gr->vertices : gr->triangles;

  bem_slpCompr = new_slp_helmholtz_bem3d(kvec, gr, q, q + 2, basis_neumann);

  bem_slp = new_slp_helmholtz_bem3d(kvec, gr, q, q + 2, basis_neumann);
  bem_dlp = new_dlp_helmholtz_bem3d(kvec, gr, q, q + 2, basis_neumann,basis_dirichlet, exterior ? 0.5 : -0.5);

  rootn = build_bem3d_cluster(bem_slp, clf, basis_neumann);
  rootd = build_bem3d_cluster(bem_dlp, clf, basis_dirichlet);

  brootV = build_nonstrict_block(rootn, rootn, &eta, admissible_max_cluster);
  brootKM = build_nonstrict_block(rootn, rootd, &eta, admissible_max_cluster);

  V = build_from_block_hmatrix(brootV, 0);
  KM = build_from_block_hmatrix(brootKM, 0);

  VfullCompr = new_amatrix(nn, nn);
  Vfull = new_amatrix(nn, nn);
  KMfull = new_amatrix(nn, nd);

  printf("----------------- filling compr matrix ------------------------\n");
//  bem_slpCompr->nearfield(NULL, NULL, bem_slpCompr, false, VfullCompr);
  assemble_ll_near_bem3d(NULL, NULL, bem_slpCompr, false, VfullCompr, slp_kernel_helmholtzbem3d_compr);
  printf("----------------- compr matrix filled ------------------------\n");

  bem_slp->nearfield(NULL, NULL, bem_slp, false, Vfull);
  printf("----------------- full matrix filled ------------------------\n");
  bem_dlp->nearfield(NULL, NULL, bem_dlp, false, KMfull);
  m = 4;
  setup_hmatrix_aprx_inter_row_bem3d(bem_slp, rootn, rootn, brootV, m);
  setup_hmatrix_aprx_inter_row_bem3d(bem_dlp, rootn, rootd, brootKM, m);


  eps_solve = 1.0e-12;
  steps = 1000;

  eval.V = V;
  eval.KM = KM;
  eval.eta = bem_slp->k;
  assemble_bem3d_hmatrix(bem_slp, brootV, (phmatrix) V);
  assemble_bem3d_hmatrix(bem_dlp, brootKM, KM);
  errorV = norm2diff_amatrix_hmatrix((phmatrix) V, Vfull) / norm2_amatrix(Vfull);
  printf("rel. error V       : %.5e\n", errorV);
  errorV = norm2diff_amatrix(VfullCompr, Vfull) / norm2_amatrix(Vfull);
  printf("rel. error VCompr  : %.5e\n", errorV);
  errorKM = norm2diff_amatrix_hmatrix(KM, KMfull) / norm2_amatrix(KMfull);
  printf("rel. error K%c0.5*M : %.5e\n", (exterior == true ? '+' : '-'), errorKM);
  eval.Vtype = HMATRIX;
  eval.KMtype = HMATRIX;

  evalf.V = Vfull;
  evalf.KM = KMfull;
  evalf.eta = bem_slp->k;
  evalf.Vtype = AMATRIX;
  evalf.KMtype = AMATRIX;


  hdata.kvec = bem_slp->kvec;
  hdata.source = allocreal(3);
  hdata.source[0] = 0.0, hdata.source[1] = 0.0, hdata.source[2] = 0.2;

  x = new_avector(Vfull->cols);
  b = new_avector(KMfull->rows);

  if (basis_dirichlet == BASIS_LINEAR_BEM3D) {
    integrate_bem3d_linear_avector(bem_dlp, rhs, b, (void *) &hdata);
  }
  else {
    integrate_bem3d_const_avector(bem_dlp, rhs, b, (void *) &hdata);
  }

  solve_gmres_bem3d(HMATRIX, (void *) &eval, b, x, eps_solve, steps);

  xf = new_avector(Vfull->cols);
  solve_gmres_bem3d(AMATRIX, (void *) &evalf, b, xf, eps_solve, steps);

  printf("apsoidjf\n");

  FILE *gnuplotVec = fopen("gpv.dat", "w");
//  fprintf(gnuplot, "plot '-'\n");
  printf("faosidf\n");
  i = 1;
  sum = 0;
  elmt = getentry_amatrix(Vfull,i,0)*getentry_avector(xf,0);
  printf("oaisufhddf\n");
  for (j = 0; j < Vfull->cols; j++) {
     elmt = getentry_amatrix(Vfull,j,i)*getentry_avector(xf,j);
//     elmt = getentry_amatrix(Vfull,i,j)*getentry_avector(xf,j);
     sum += elmt;
//     fprintf(gnuplotVec, "%i %g\n", j, elmt);
     fprintf(gnuplotVec, "%i %g\n", j, creal(elmt));
//     fprintf(gnuplotVec, "%i %g\n", j, abs(elmt));
  }
  printf("aopsufhd\n");
//  fprintf(gnuplotVec, "e\n");
//  fflush(gnuplotVec);
  fclose(gnuplotVec);
  sys = system("gnuplot asyCompr/gnuPlVec");
  printf("a0soijfdf\n");
  elmt = getentry_avector(b,i);
//  printf("i= %i, sum = %g, b[i] = %g, showing A[i,:]x[:]...\n", i, sum, elmt);
  printf("i= %i, sum = %g+i%g, b[i] = %g+i%g, showing A[i,:]x[:]...\n", i, creal(sum), cimag(sum), creal(elmt), cimag(elmt));
//  sys += system("evince gpv.eps");
  printf("pouiahdf\n");

  sum = 0.0;
  norb = 0.0;
  norerr = 0.0;
  for(i=0; i < Vfull->rows; i++) {
     elmt = -getentry_avector(b,i);
     norb += pow(cabs(elmt),2.0);
     for (j = 0; j < Vfull->cols; j++) {
	elmt += getentry_amatrix(Vfull,j,i)*getentry_avector(xf,j);
//	elmt += getentry_amatrix(Vfull,i,j)*getentry_avector(xf,j);
     }
     sum += pow(cabs(elmt),2.0);
//     printf("%g+i%g = elmt, norb= %g+i%g\n", creal(elmt), cimag(elmt), creal(norb), cimag(norb) );
  }
  printf("rows=%i, cols=%i.\n", Vfull->rows, Vfull->cols);
  printf("norb=%g, norerr=%g, sys=%i, norerr=%g.\n", sqrt(cabs(norb)), sqrt(cabs(sum)), sys, sqrt(cabs(norerr)));

//  tst = b;
//  tst = 0;
  tst = new_avector(KMfull->rows);
  copy_avector(b,tst);
  addeval_amatrix_avector(-1.0, Vfull, xf, tst);
  printf("norb with norm2=%g, norerr with addeval=%g \n.", norm2_avector(b), norm2_avector(tst) );


  error_solve = max_rel_outer_error(bem_slp, &hdata, x, rhs, basis_neumann);
  printf("max. rel. error Hmat: %.5e \n", error_solve);
  add_avector(-1.0, xf, x); // x is overwritten by x-1.0*xf
  errSol = norm2_avector(x);
  printf("errSol Hmat= %.5e \n", errSol);


  evalfC.V = VfullCompr;
  evalfC.KM = KMfull;
  evalfC.eta = bem_slp->k;
  evalfC.Vtype = AMATRIX;
  evalfC.KMtype = AMATRIX;
  xfc = new_avector(VfullCompr->cols);
  solve_gmres_bem3d(AMATRIX, (void *) &evalfC, b, xfc, eps_solve, steps);


  error_solve = max_rel_outer_error(bem_slp, &hdata, xfc, rhs, basis_neumann);
  printf("max. rel. error compr : %.5e \n", error_solve);
  add_avector(-1.0, xf, xfc); // xfc is overwritten by xfc-1.0*xf
//  add_avector(-1.0, x, xfc); // xfc is overwritten by xfc-1.0*x
  errSol = norm2_avector(xfc);
  printf("errSol compr= %.5e \n", errSol);


  del_avector(x);
  del_avector(xf);
  del_avector(xfc);
  del_avector(b);
  freemem(hdata.source);

  del_block(brootV);
  del_block(brootKM);
  freemem(rootn->idx);
  freemem(rootd->idx);
  del_cluster(rootn);
  del_cluster(rootd);
  del_amatrix(Vfull);
  del_amatrix(KMfull);
  del_helmholtz_bem3d(bem_slp);
  del_helmholtz_bem3d(bem_dlp);


  del_surface3d(gr);
  del_macrosurface3d(mg);
  (void) printf("%u matrices and %u vectors still active\n", getactives_amatrix(), getactives_avector());
  uninit_h2lib();
  return 0;
}



#include "basic.h"
#include "krylov.h"
#include "helmholtzbem3d.h"
#include "validation.c"

#define IS_IN_RANGE(a, b, c) (((a) <= (b)) && ((b) <= (c)))

const bool* printed;// = &false;

int
main(int argc, char **argv)
{
  pmacrosurface3d mg;
  psurface3d gr;
  uint      n, q, clf;
  real      eta; //, elmt, sum;
  field     kvec[3];
  
  struct _eval_A eval;
  pavector  x, b;
  helmholtz_data hdata;
  real      error_solve, eps_solve;
  uint      steps;
  boundary_func3d rhs = (boundary_func3d) rhs_dirichlet_point_helmholtzbem3d;

  pbem3d    bem_slp, bem_dlp;
  pcluster  rootn, rootd;
  pblock    brootV, brootKM;
  phmatrix  V, KM;
  uint      m, mode;
  uint      l;
  real      delta;
  real      eps_aca;

  pclusterbasis Vrb, Vcb, KMrb, KMcb;
  ph2matrix V2, KM2;

  basisfunctionbem3d basis_neumann;
  basisfunctionbem3d basis_dirichlet;
  bool exterior;
  matrixtype mattype;

  init_h2lib(&argc, &argv);

  mode = 1;

//  n = 512;
//  kvec[0] = 2.0, kvec[1] = 0.0, kvec[2] = 0.0;
  if(argc != 3) { printf("Please do $ ./a.out N k"); }
  if (sscanf(argv[1], "%i", &n)!=1) { printf("error - not an integer"); }
  if (sscanf(argv[2], "%lf", &eta)!=1) { printf("error - not a float"); }
  kvec[0] = eta, kvec[1] = 0.0, kvec[2] = 0.0;

  q = 2;
  eta = 1.0;

  mg = new_sphere_macrosurface3d();
  gr = build_from_macrosurface3d_surface3d(mg, REAL_SQRT(n * 0.125));
  n = gr->triangles;

//  write_surface3d(gr, "../H2Lib_trunk/sphere.tri");
  printf("Testing unit sphere with %d triangles and %d vertices\n", gr->triangles, gr->vertices);
  clf = 16;

  basis_neumann = BASIS_LINEAR_BEM3D;
  basis_dirichlet = BASIS_LINEAR_BEM3D;
  exterior = true;

  bem_slp = new_slp_helmholtz_bem3d(kvec, gr, q, q + 2, basis_neumann);
  bem_dlp = new_dlp_helmholtz_bem3d(kvec, gr, q, q + 2, basis_neumann,basis_dirichlet, exterior ? 0.5 : -0.5);

  rootn = build_bem3d_cluster(bem_slp, clf, basis_neumann);
  rootd = build_bem3d_cluster(bem_dlp, clf, basis_dirichlet);

  eps_solve = 1.0e-12;
  steps = 1000;

  eval.eta = bem_slp->k;

  hdata.kvec = bem_slp->kvec;
  hdata.source = allocreal(3);
  hdata.source[0] = 0.0, hdata.source[1] = 0.0, hdata.source[2] = 0.2;

  if (mode == 0) { // ll exterior Hmatrix interpolation row
    mattype = HMATRIX;
    brootV = build_nonstrict_block(rootn, rootn, &eta, admissible_max_cluster);
    brootKM = build_nonstrict_block(rootn, rootd, &eta, admissible_max_cluster);

    V = build_from_block_hmatrix(brootV, 0);
    KM = build_from_block_hmatrix(brootKM, 0);
    m = 4;
    setup_hmatrix_aprx_inter_row_bem3d(bem_slp, rootn, rootn, brootV, m);
    setup_hmatrix_aprx_inter_row_bem3d(bem_dlp, rootn, rootd, brootKM, m);

    assemble_bem3d_hmatrix(bem_slp, brootV, (phmatrix) V);
    assemble_bem3d_hmatrix(bem_dlp, brootKM, KM);
    eval.V = V;
    eval.KM = KM;
    eval.Vtype = HMATRIX;
    eval.KMtype = HMATRIX;
//  test_system(HMATRIX, "Interpolation row", Vfull, KMfull, brootV, bem_slp, V, brootKM, bem_dlp, KM, basis_neumann, basis_dirichlet, exterior, error_min, error_max);
    x = new_avector(getcols_hmatrix(V));
    b = new_avector(getrows_hmatrix(KM));
  } else{ // ll exterior H2matrix (GreenHybridOrtho)
    mattype = H2MATRIX;
    brootV = build_strict_block(rootn, rootn, &eta, admissible_max_cluster);
    brootKM = build_strict_block(rootn, rootd, &eta, admissible_max_cluster);
//printf("apsoifdj\n");
    Vrb = build_from_cluster_clusterbasis(rootn);
    Vcb = build_from_cluster_clusterbasis(rootn);
    KMrb = build_from_cluster_clusterbasis(rootn);
    KMcb = build_from_cluster_clusterbasis(rootd);

    V2 = build_from_block_h2matrix(brootV, Vrb, Vcb);
    KM2 = build_from_block_h2matrix(brootKM, KMrb, KMcb);
//printf("fdj\n");
    m = 2;
    l = 1;
    delta = 1.0;
    eps_aca = 5.0e-3;
//printf("9oaisufhd\n");
    if (false) {
	setup_h2matrix_aprx_greenhybrid_ortho_bem3d(bem_slp, Vrb, Vcb, brootV, m, l, delta, eps_aca, build_bem3d_cube_quadpoints);
	setup_h2matrix_aprx_greenhybrid_ortho_bem3d(bem_dlp, KMrb, KMcb, brootKM, m, l, delta, eps_aca, build_bem3d_cube_quadpoints);
    } else {
	setup_h2matrix_aprx_inter_bem3d(bem_slp, Vrb, Vcb, brootV, m);
	setup_h2matrix_aprx_inter_bem3d(bem_dlp, KMrb, KMcb, brootKM, m);
    }
// test_system
    assemble_bem3d_h2matrix_row_clusterbasis(bem_slp, V2->rb);
//printf("o\n");
    assemble_bem3d_h2matrix_col_clusterbasis(bem_slp, V2->cb);
//printf("fho\n");
    assemble_bem3d_h2matrix(bem_slp, brootV, V2);
//printf("oiuholiuhho\n");
    assemble_bem3d_h2matrix_row_clusterbasis(bem_dlp, KM2->rb);
    assemble_bem3d_h2matrix_col_clusterbasis(bem_dlp, KM2->cb);
    assemble_bem3d_h2matrix(bem_dlp, brootKM, KM2);

    eval.V = V2;
    eval.KM = KM2;
    eval.Vtype = H2MATRIX;
    eval.KMtype = H2MATRIX;
//  test_system(H2MATRIX, "Greenhybrid ortho", Vfull, KMfull, brootV, bem_slp, V2, brootKM, bem_dlp, KM2, basis_neumann, basis_dirichlet, exterior, error_min, error_max);
//    x = new_avector(getcols_h2matrix(V2));h2->cb->t->size
//printf("apsoifdj\n");
    x = new_avector(V2->cb->t->size);
    b = new_avector(KM2->rb->t->size);
  }


  if (basis_dirichlet == BASIS_LINEAR_BEM3D) {
    integrate_bem3d_linear_avector(bem_dlp, rhs, b, (void *) &hdata);
  }
  else {
    integrate_bem3d_const_avector(bem_dlp, rhs, b, (void *) &hdata);
  }
//printf("9suydfhgoasiuhd\n");
  solve_gmres_bem3d(mattype, (void *) &eval, b, x, eps_solve, steps);
//printf("apsoifdj\n");
  error_solve = max_rel_outer_error(bem_slp, &hdata, x, rhs, basis_neumann);
  printf("max. rel. error Hmat: %.5e \n", error_solve);


  del_avector(x);
  del_avector(b);
  freemem(hdata.source);

  del_block(brootV);
  del_block(brootKM);
  freemem(rootn->idx);
  freemem(rootd->idx);

  del_h2matrix(V2);
  del_h2matrix(KM2);

  del_cluster(rootn);
  del_cluster(rootd);
  del_helmholtz_bem3d(bem_slp);
  del_helmholtz_bem3d(bem_dlp);


  del_surface3d(gr);
  del_macrosurface3d(mg);
  (void) printf("%u matrices and %u vectors still active\n", getactives_amatrix(), getactives_avector());
  uninit_h2lib();
  return 0;
}



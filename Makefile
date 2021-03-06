
# ------------------------------------------------------------
# Components of the main library
# ------------------------------------------------------------

H2LIB_CORE0 = \
	Library/basic.c \
	Library/settings.c \
	Library/parameters.c \
	Library/opencl.c

H2LIB_CORE1 = \
	Library/avector.c \
	Library/realavector.c \
	Library/amatrix.c \
	Library/factorizations.c \
	Library/eigensolvers.c \
	Library/sparsematrix.c \
	Library/sparsepattern.c \
	Library/gaussquad.c \
	Library/krylov.c

H2LIB_CORE2 = \
	Library/cluster.c \
	Library/clustergeometry.c \
	Library/block.c \
	Library/clusterbasis.c \
	Library/clusteroperator.c \
	Library/uniform.c \
	Library/h2matrix.c \
	Library/rkmatrix.c \
	Library/hmatrix.c

H2LIB_CORE3 = \
	Library/truncation.c \
	Library/harith.c \
	Library/harith2.c \
	Library/hcoarsen.c \
	Library/h2compression.c \
	Library/h2update.c \
	Library/h2arith.c \
	Library/aca.c

H2LIB_SIMPLE = 

H2LIB_FEM = \
	Library/tri2d.c \
	Library/tri2dp1.c \
	Library/tet3d.c \
	Library/tet3dp1.c\
	Library/ddcluster.c


H2LIB_BEM = \
	Library/curve2d.c \
	Library/singquad1d.c \
	Library/bem2d.c \
	Library/laplacebem2d.c \
	Library/surface3d.c \
	Library/macrosurface3d.c \
	Library/singquad2d.c \
	Library/bem3d.c \
	Library/oclbem3d.c \
	Library/laplacebem3d.c \
	Library/laplaceoclbem3d.c \
	Library/helmholtzbem3d.c \
	Library/helmholtzoclbem3d.c

SOURCES_libh2 := \
	$(H2LIB_CORE0) \
	$(H2LIB_CORE1) \
	$(H2LIB_CORE2) \
	$(H2LIB_CORE3) \
	$(H2LIB_SIMPLE) \
	$(H2LIB_FEM) \
	$(H2LIB_BEM)

HEADERS_libh2 := $(SOURCES_libh2:.c=.h)

OBJECTS_libh2 := $(SOURCES_libh2:.c=.o)

DEPENDENCIES_libh2 := $(SOURCES_libh2:.c=.d)

# ------------------------------------------------------------
# Test programs
# ------------------------------------------------------------

SOURCES_stable := \
	Tests/test_amatrix.c \
	Tests/test_eigen.c \
	Tests/test_hmatrix.c \
	Tests/test_h2matrix.c \
	Tests/test_krylov.c \
	Tests/test_laplacebem2d.c \
	Tests/test_laplacebem3d.c \
	Tests/test_laplacebem3d_ocl.c \
	Tests/test_helmholtzbem3d.c \
	Tests/test_helmholtzbem3d_ocl.c \
	Tests/test_h2compression.c \
	Tests/test_tet3d.c \
	Tests/test_tri2d.c \
	Tests/test_ddcluster.c
#	asyCompr/le1ir.c

SOURCES_tests = $(SOURCES_stable) \

OBJECTS_tests := \
	$(SOURCES_tests:.c=.o)

DEPENDENCIES_tests := \
	$(SOURCES_tests:.c=.d)

PROGRAMS_tests := \
	$(SOURCES_tests:.c=)

# ------------------------------------------------------------
# All files
# ------------------------------------------------------------

SOURCES := \
	$(SOURCES_libh2) \
	$(SOURCES_tests)

HEADERS := \
	$(HEADER_libh2)

OBJECTS := \
	$(OBJECTS_libh2) \
	$(OBJECTS_tests)

DEPENDENCIES := \
	$(DEPENDENCIES_libh2) \
	$(DEPENDENCIES_tests)

PROGRAMS := \
	$(PROGRAMS_tests)

# ------------------------------------------------------------
# Standard target
# ------------------------------------------------------------

all: programs

# ------------------------------------------------------------
# Build configuration
# ------------------------------------------------------------

$(OBJECTS): options.inc
include options.inc

# ------------------------------------------------------------
# System-dependent parameters (e.g., name of compiler)
# ------------------------------------------------------------

$(OBJECTS): system.inc
include system.inc

# ------------------------------------------------------------
# System-independent configuration (e.g., variants of algorithms)
# ------------------------------------------------------------

ifdef HARITH_RKMATRIX_QUICK_EXIT
CFLAGS += -DHARITH_RKMATRIX_QUICK_EXIT
endif

ifdef HARITH_AMATRIX_QUICK_EXIT
CFLAGS += -DHARITH_AMATRIX_QUICK_EXIT
endif

# ------------------------------------------------------------
# Rules for test programs
# ------------------------------------------------------------

programs: $(PROGRAMS_tests)

$(PROGRAMS_tests): %: %.o
ifdef BRIEF_OUTPUT
	@echo Linking $@
	@$(CC) $(LDFLAGS) -Wl,-L,.,-R,. $< -o $@ -lh2 -lm $(LIBS) 
else
	$(CC) $(LDFLAGS) -Wl,-L,.,-R,. $< -o $@ -lh2 -lm $(LIBS)
endif

$(PROGRAMS_tests) $(PROGRAMS_tools): libh2.a
#$(PROGRAMS_tests) $(PROGRAMS_tools) asyCompr: libh2.a

$(OBJECTS_tests): %.o: %.c
ifdef BRIEF_OUTPUT
	@echo Compiling $<
	@$(GCC) -MT $@ -MM -I Library $< > $(<:%.c=%.d)
	@$(CC) $(CFLAGS) -I Library -c $< -o $@
else
	@$(GCC) -MT $@ -MM -I Library $< > $(<:%.c=%.d)
	$(CC) $(CFLAGS) -I Library -c $< -o $@
endif

-include $(DEPENDENCIES_tests) $(DEPENDENCIES_tools)
$(OBJECTS_tests): Makefile

# ------------------------------------------------------------
# Rules for the Doxygen documentation
# ------------------------------------------------------------

doc:
	doxygen Doc/Doxyfile

# ------------------------------------------------------------
# Rules for the main library
# ------------------------------------------------------------

libh2.a: $(OBJECTS_libh2)
ifdef BRIEF_OUTPUT
	@echo Building $@
	@$(AR) $(ARFLAGS) $@ $(OBJECTS_libh2)
else
	$(AR) $(ARFLAGS) $@ $(OBJECTS_libh2)
endif

$(OBJECTS_libh2): %.o: %.c
ifdef BRIEF_OUTPUT
	@echo Compiling $<
	@$(GCC) -MT $@ -MM $< > $(<:%.c=%.d)
	@$(CC) $(CFLAGS) -c $< -o $@
else
	@$(GCC) -MT $@ -MM $< > $(<:%.c=%.d)
	$(CC) $(CFLAGS) -c $< -o $@
endif

-include $(DEPENDENCIES_libh2)
$(OBJECTS_libh2): Makefile

# ------------------------------------------------------------
# Useful additions
# ------------------------------------------------------------

.PHONY: clean cleandoc programs indent

clean:
	$(RM) -f $(OBJECTS) $(DEPENDENCIES) $(PROGRAMS) libh2.a

cleandoc:
	$(RM) -rf Doc/html Doc/latex

indent:
	indent -bap -br -nce -cdw -npcs \
	  -di10 -nbc -brs -blf -i2 -lp \
	  -T amatrix -T pamatrix -T pcamatrix \
	  -T avector -T pavector -T pcavector \
	  -T cluster -T pcluster -T pccluster \
	  -T block -T pblock -T pcblock \
	  -T rkmatrix -T prkmatrix -T pcrkmatrix \
	  -T hmatrix -T phmatrix -T pchmatrix \
	  -T uniform -T puniform -T pcuniform \
	  -T h2matrix -T ph2matrix -T pch2matrix \
	  $(SOURCES)

coverage:
	mkdir Coverage > /dev/null 2>&1; \
	lcov --base-directory . --directory . --capture \
	--output-file Coverage/coverage.info && \
	genhtml -o Coverage Coverage/coverage.info

cleangcov:
	$(RM) -rf Library/*.gcov Library/*.gcda Library/*.gcno \
	Tests/*.gcov Tests/*.gcda Tests/*.gcno;
	$(RM) -rf Coverage


# ------------------------------------------------------------
# asyCompr Test programs
# ------------------------------------------------------------

#SOURCES_sasyCompr := \
#	Tests/test_amatrix.c \
#	asyCompr/le1ir.c

#SOURCES_asyCompr = $(SOURCES_sasyCompr) \

#OBJECTS_asyCompr := \
#	$(SOURCES_asyCompr:.c=.o)

#DEPENDENCIES_asyCompr := \
#	$(SOURCES_asyCompr:.c=.d)

#PROGRAMS_asyCompr := \
#	$(SOURCES_asyCompr:.c=)

# --------------------------------------------------------
# Asymptotic compression
# --------------------------------------------------------

#asyCompr: %: %.o
#	$(CC) $(LDFLAGS) -Wl,-L,.,-R,. $< -o $@ -lh2 -lm $(LIBS) asyCompr/le1ir.c -ILibrary
#asyCompr:
#	$(CC) $(LDFLAGS) -lh2 -lm $(LIBS) asyCompr/le1ir.c -ILibrary Library/*.o
#.PHONY: asyCompr

#CPA = gcc -Wall -O3 -march=native -funroll-loops -funswitch-loops -DUSE_COMPLEX -DUSE_BLAS -DUSE_CAIRO -I/usr/include/cairo -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include -I/usr/include/pixman-1 -I/usr/include/freetype2 -I/usr/include/libpng12   -DHARITH_RKMATRIX_QUICK_EXIT -DHARITH_AMATRIX_QUICK_EXIT -I Library -c
CPA = -Wall -O3 -march=native -funroll-loops -funswitch-loops -DUSE_COMPLEX -DUSE_BLAS -DUSE_CAIRO -I/usr/include/cairo -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include -I/usr/include/pixman-1 -I/usr/include/freetype2 -I/usr/include/libpng12   -DHARITH_RKMATRIX_QUICK_EXIT -DHARITH_AMATRIX_QUICK_EXIT -I Library -Wno-int-to-pointer-cast
LIA = -lh2 -lm  -llapack -lblas -lgfortran -lcairo -lm

asyCompr:
	@gcc $(CPA) -c asyCompr/validation.c -Wno-unused-function
#	@gcc $(CPA) -c asyCompr/tryCompr.c
#	@gcc -Wl,-L,.,-R,. $(CPA) tryCompr.o validation.o $(LIA)
#	@gcc $(CPA) -c asyCompr/tryCompr.c validation.o
#	@gcc -Wl,-L,.,-R,. $(CPA) tryCompr.o $(LIA)
	@gcc -Wl,-L,.,-R,. $(CPA) asyCompr/tryCompr.c $(LIA)
#	$(CPA) asyCompr/validation.c -o asyCompr/validation.o#
#	$(CPA) asyCompr/tryCompr.c -o asyCompr/tryCompr.o
#	gcc  -Wl,-L,.,-R,. asyCompr/tryCompr.o asyCompr/validation.o -o tryCompr $(LIA)


speed:
	@gcc $(CPA) -c asyCompr/validation.c -Wno-unused-function
	@gcc -Wl,-L,.,-R,. $(CPA) asyCompr/speedH2lib.c $(LIA)

aOld:
#	$(CPA) asyCompr/le1ir.c -o asyCompr/le1ir.o
#	gcc  -Wl,-L,.,-R,. asyCompr/le1ir.o -o le1ir $(LIA)
	gcc -Wall -O3 -march=native -funroll-loops -funswitch-loops -DUSE_COMPLEX -DUSE_BLAS -DUSE_CAIRO -I/usr/include/cairo -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include -I/usr/include/pixman-1 -I/usr/include/freetype2 -I/usr/include/libpng12   -DHARITH_RKMATRIX_QUICK_EXIT -DHARITH_AMATRIX_QUICK_EXIT -I Library -c asyCompr/le1ir.c -o asyCompr/le1ir.o
	gcc  -Wl,-L,.,-R,. asyCompr/le1ir.o -o le1ir -lh2 -lm  -llapack -lblas -lgfortran -lcairo -lm
#	make clean
.PHONY: asyCompr



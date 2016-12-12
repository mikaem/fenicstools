cd /home/travis/miniconda2/envs/_test/lib/python2.7/site-packages/dolfin/cpp
python -c "import _common"
cd $SRC_DIR/tests
py.test -v
#mpirun -np 2 py.test -v

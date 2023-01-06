export PYTHONPATH="/Users/yangjunjie/opt/pyscf/pyscf-main/"
export PYTHONPATH="/Users/yangjunjie/opt/cqcpy/cqcpy-master/:$PYTHONPATH"
export PYTHONPATH="/Users/yangjunjie/opt/wick/wick-dev/:$PYTHONPATH"
export PYTHONPATH="/Users/yangjunjie/opt/epcc/epcc-master:$PYTHONPATH"
export PYTHONPATH="/Users/yangjunjie/work/cceqs/:$PYTHONPATH"
echo $PYTHONPATH;

python -c "import pyscf; print(pyscf.__file__)"
python -c "import cqcpy; print(cqcpy.__file__)"
python -c "import wick; print(wick.__file__)"
python -c "import epcc; print(epcc.__file__)"

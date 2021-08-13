
# set env varible PATH to remove 'node_module/.bin'

cp -r ./streamPredictor.py $1
cp -r ./starry $1

cd $1

find . -name '*.pyc' -type f -print -exec rm {} \;
python3 -O -m compileall .
find . -name '*.pyc' -exec rename 's/.cpython-3..opt-1//' {} \;
find . -name '*.pyc' -execdir mv {} .. \;
find . -name '*.py' -type f -print -exec rm {} \;
find . -name '__pycache__' -exec rmdir {} \;

#rm ./*.ipynb
#rm ./*.txt
#rm ./*.md
#rm -r ./configs

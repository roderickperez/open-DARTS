# rename whl: replace the "linux" suffix with "manylinux2014" to make uploadable to pypi
for fname in `ls ./dist/*whl`
do
    newfname=`echo $fname | sed "s|linux|manylinux2014|g"`
    mv $fname $newfname
done

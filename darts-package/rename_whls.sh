for fname in `ls ./dist/*whl`
do
    newfname=`echo $fname | sed "s|linux|manylinux2014|g"`
    mv $fname $newfname
done

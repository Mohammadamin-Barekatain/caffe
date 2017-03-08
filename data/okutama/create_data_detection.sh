root_dir=/home/amin/Amin-SSD/caffe

cd $root_dir

redo=true
data_root_dir="/"
dataset_name="okutama"
mapfile="$root_dir/data/$dataset_name/labelmap_SDD_detection.prototxt"
anno_type="detection"
label_type="xml"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if $redo
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in trainval test
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-type=$label_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$subset"_Detection".txt /home/amin/$db/$dataset_name"_Detection_"$subset"_"$db examples/$dataset_name 2>&1 | tee $root_dir/data/$dataset_name/$subset.log
done


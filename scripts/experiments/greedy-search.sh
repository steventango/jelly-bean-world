# loop from 0 to 29
for i in {0..29}
do
    ./bin/greedy_visual_agent $i &
done
wait

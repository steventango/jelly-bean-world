seeds=(16 27 77 10 55 84 99 53 54 8 34 90 70 43 11 26 67 71 42 68 33 15 37 17 69 7 78 14 94 87129 156 145 112 119)
for i in ${seeds[*]};
do
    ./bin/greedy_visual_agent $i &
done
wait

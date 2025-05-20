for step in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
    rm weights/*

    python mnist_exp.py --epoch 50 --save "weights/clean_mlp"

    python linear_probe.py --weight "weights/clean_mlp_final.pth" >> mlp_full_$step.txt

    python mnist_exp.py --noisy --epoch 50 --save "weights/noisy_mlp"

    python linear_probe_noisy.py --weight "weights/noisy_mlp_final.pth" >> mlp_full_$step.txt

    python mnist_exp.py --noisy --epoch 20 --save "weights/clean2noisy_mlp" --weight "weights/clean_mlp_30.pth"

    python linear_probe_noisy.py --weight "weights/clean2noisy_mlp_final.pth" >> mlp_full_$step.txt
done
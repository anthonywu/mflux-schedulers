#!/bin/zsh -e

gen_id=${RANDOM}
prompt_file=/tmp/prompt_${gen_id}.txt
echo "macro photograph of a giant, bioluminescent jellyfish floating in the deep ocean. Its translucent bell reveals intricate internal structures, and its long, ethereal tentacles trail behind, emitting a soft, neon blue and pink light. The surrounding water is dark and inky, with tiny plankton and air bubbles catching the light. Ultra-realistic, tack sharp, high contrast, National Geographic style." > $prompt_file
output_file_linear=/tmp/image_${gen_id}_linear.png
output_file_euler_discrete=/tmp/image_${gen_id}_euler_discrete.png


mflux-generate \
    --prompt-file $prompt_file \
    --model dev \
    --steps 20 \
    --scheduler linear \
    --seed $gen_id \
    --output $output_file_linear

mflux-generate \
    --prompt-file $prompt_file \
    --model dev \
    --steps 20 \
    --scheduler mflux.contrib.schedulers.euler_discrete_scheduler.EulerDiscreteScheduler \
    --seed $gen_id \
    --output $output_file_euler_discrete

if [ -f $output_file_linear ]; then
    open $output_file_linear
fi

if [ -f $output_file_euler_discrete ]; then
    open $output_file_euler_discrete
fi

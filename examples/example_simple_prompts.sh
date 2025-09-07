#!/bin/zsh -e

# Simple prompts to test scheduler differences
gen_id=${RANDOM}

# Test 1: Simple portrait
prompt1="portrait of a cat wearing sunglasses, studio lighting"
echo $prompt1 > /tmp/prompt1_${gen_id}.txt

mflux-generate \
    --prompt-file /tmp/prompt1_${gen_id}.txt \
    --model dev \
    --steps 20 \
    --scheduler linear \
    --seed $gen_id \
    --output /tmp/cat_${gen_id}_linear.png

mflux-generate \
    --prompt-file /tmp/prompt1_${gen_id}.txt \
    --model dev \
    --steps 20 \
    --scheduler mflux.contrib.schedulers.euler_discrete_scheduler.EulerDiscreteScheduler \
    --seed $gen_id \
    --output /tmp/cat_${gen_id}_euler.png

# Test 2: Simple landscape
prompt2="mountain landscape at sunset, golden hour"
echo $prompt2 > /tmp/prompt2_${gen_id}.txt

mflux-generate \
    --prompt-file /tmp/prompt2_${gen_id}.txt \
    --model dev \
    --steps 20 \
    --scheduler linear \
    --seed $gen_id \
    --output /tmp/mountain_${gen_id}_linear.png

mflux-generate \
    --prompt-file /tmp/prompt2_${gen_id}.txt \
    --model dev \
    --steps 20 \
    --scheduler mflux.contrib.schedulers.euler_discrete_scheduler.EulerDiscreteScheduler \
    --seed $gen_id \
    --output /tmp/mountain_${gen_id}_euler.png

# Test 3: Simple object
prompt3="red apple on white background"
echo $prompt3 > /tmp/prompt3_${gen_id}.txt

mflux-generate \
    --prompt-file /tmp/prompt3_${gen_id}.txt \
    --model dev \
    --steps 20 \
    --scheduler linear \
    --seed $gen_id \
    --output /tmp/apple_${gen_id}_linear.png

mflux-generate \
    --prompt-file /tmp/prompt3_${gen_id}.txt \
    --model dev \
    --steps 20 \
    --scheduler mflux.contrib.schedulers.euler_discrete_scheduler.EulerDiscreteScheduler \
    --seed $gen_id \
    --output /tmp/apple_${gen_id}_euler.png

echo "Generated images for comparison:"
echo "Cat: /tmp/cat_${gen_id}_linear.png vs /tmp/cat_${gen_id}_euler.png"
echo "Mountain: /tmp/mountain_${gen_id}_linear.png vs /tmp/mountain_${gen_id}_euler.png"
echo "Apple: /tmp/apple_${gen_id}_linear.png vs /tmp/apple_${gen_id}_euler.png"

# Open all generated images
for file in /tmp/cat_${gen_id}_*.png /tmp/mountain_${gen_id}_*.png /tmp/apple_${gen_id}_*.png; do
    if [ -f "$file" ]; then
        open "$file"
    fi
done

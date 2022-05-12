#!/bin/bash
script_dir=$(dirname "$0")
current_scratchpad=$(ls -1 -t $script_dir/data/scratchpad/ | head -n 1)
current_images_folder="$script_dir/data/scratchpad/$current_scratchpad/images"
last_generated_image=$(ls -1 -t $current_images_folder/*_g.png | head -n 1 | xargs basename)
last_images_prefix=${last_generated_image%_g*}

echo $current_images_folder/$last_images_prefix

cacaview $current_images_folder/$last_images_prefix*.png


tricks=(fgm eight_bit)

# # usual fgm\eight_bit
# ids=(6 5)
# for i in 0 1; do
#     nohup python -u center_controller.py \
#         --gpu=${ids[i]} \
#         --pretrained_model=../../hf-plm/bert-base-chinese \
#         --out_dir=saved/usual/${tricks[i]} \
#         --trick_name=${tricks[i]} \
#         --task_config=default_configs/text_clf_smp2020_ewect_usual.yaml >text_clf_smp2020_ewct_usual_${tricks[i]}-1.log 2>&1 &
# done

# # virus fgm\eight_bit
# ids=(1 0)
# for i in 0 1; do
#     nohup python -u center_controller.py \
#         --gpu=${ids[i]} \
#         --pretrained_model=../../hf-plm/bert-base-chinese \
#         --out_dir=saved/virus/${tricks[i]} \
#         --trick_name=${tricks[i]} \
#         --task_config=default_configs/text_clf_smp2020_ewect_virus.yaml >text_clf_smp2020_ewct_virus_${tricks[i]}-1.log 2>&1 &
# done

# usual\virus base
files=(usual virus)
ids=(1 2)
for i in 0 1; do
    nohup python -u center_controller.py \
        --gpu=${ids[i]} \
        --pretrained_model=../../hf-plm/bert-base-chinese \
        --out_dir=saved/${files[i]}/base \
        --task_config=default_configs/text_clf_smp2020_ewect_${files[i]}.yaml >text_clf_smp2020_ewct_${files[i]}_base-1.log 2>&1 &
done


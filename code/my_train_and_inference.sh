
# source YOUR DOCKER


# The Shellparameter that controls the mainprocess
FLAG=1
# The hyperparameter you need setting: 1.MODEL_NAME, 2.Experiments_setting, 3.dataset, 4.accumulations, 5.graphics_card

# select basemodel
# MODEL_NAME='ChatGLM'
# MODEL_NAME='ChatGLM2'
# MODEL_NAME='LLaMA'
MODEL_NAME='LLaMA2'
# MODEL_NAME='Bloom-560m'

# select the experiment's model
# Experiments_setting='test'
# Experiments_setting='zero_shot'
# Experiments_setting='few_shot'
Experiments_setting='lora'
# Experiments_setting='all_parameters'

# select the dataset
# dataset='iemocap'
dataset='meld'
# dataset='EmoryNLP'

# select the historical window for dataset
# LLaMA 's context = 1024 is enough for almost dataset, except for iemocap.
# IEMOCAP has very long conversation sample, 
# the historical window is designed for this kind of long conversation.
historical_window=20

# set the accumulation and card when backwarding and inferring
accumulations=8
graphics_card=2
BS=$((accumulations * graphics_card))

# parameter that determines whether the speaker_identification task is add to train stage
# meanwhile the speaker_identification loss is also added to the total loss 
# (actually another same next token prediction loss)
# speaker_task has three options[True, True_mixed, None]
# speaker_task='True' 
# speaker_task='True_mixed'
speaker_task='True'
echo "speaker_task: ${speaker_task}"
# True means storing the processed data separately (two Stage training)
# True_mixed means storing the processed data Unifiedly (One stage training)
# None means no speaker identification task in main task (only processed in window mode)

# domain_base='True'
domain_base='False'
echo "domain_base: ${domain_base}"



# parameter that determines whether the emotion_prediction task is added to train stage, 
# meanwhile the KL divergence is added to the total loss
# emotion_prediction='True'
emotion_prediction='True'
echo "emotion_prediction: ${emotion_prediction}"

data_percent=1.0    # 1
# data_percent=0.5    # 1/2
# data_percent=0.25   # 1/4
# data_percent=0.125  # 1/8
# data_percent=0.0625 # 1/16 
# data_percent=0.03125 # 1/32 
# data_percent=0.015625 # 1/64 
echo "data_percent: ${data_percent}"


# Notes: bloom-560 is convenient for debugging
case ${MODEL_NAME} in
'ChatGLM'|'ChatGLM2'|'LLaMA'|'LLaMA2'|'Bloom-560m')
    case ${Experiments_setting} in
    'zero_shot'|'few_shot'|'lora'|'all_parameters')
        case ${dataset} in
        'iemocap'|'meld'|'EmoryNLP')
            echo "******************************************************************************************"
            echo "All parameters are valid."
            echo "The dataset you have selected is: ${dataset} !"
            echo "The base model you have selected is ${MODEL_NAME}!"
            echo "The model's SFT method you have selected: ${Experiments_setting}!"
            echo "******************************************************************************************"
            ;;
        *)
            echo "The dataset parameter is invalid. CHECK IT OUT!"
            FLAG=0
            ;;
        esac
        ;;
    *)
        echo "The Experiments_setting parameter is invalid. CHECK IT OUT!"
        FLAG=0
        ;;
    esac
    ;;
*)
    echo "The MODEL_NAME parameter is invalid. CHECK IT OUT!"
    FLAG=0
    ;;
esac

# MAX_LENGTH=1200 

# DATA_PATH=$(python ./code/data_process.py --dataset ${dataset} \
#     --historical_window ${historical_window} \
#     --speaker_task ${speaker_task} \
#     --domain_base ${domain_base} \
#     --emotion_prediction ${emotion_prediction})
# if [ $? -eq 0 ]; then
#     echo "******************************************************************************************"
#     echo "Data procession has executed successfully !"
#     echo "******************************************************************************************"

# else
#     echo "Data procession script encountered an error."
# fi

if [ ${dataset} = 'iemocap' ]    
then
    MAX_LENGTH=1200
elif [ ${dataset} = 'meld' ]
then
    MAX_LENGTH=1024
elif [ ${dataset} = 'EmoryNLP' ]
then
    MAX_LENGTH=1024
else
    echo "Your choose is not in MY candidations! Please check your Model name!"
fi
echo "******************************************************************************************"
echo "Your choose ${dataset}! The max_context_length will be set as ${MAX_LENGTH}!"
echo "******************************************************************************************"


if [ ${MODEL_NAME} = 'LLaMA2' ]
then
    MODEL_PATH='/home/fock/code/InstructERC/LLM_bases/LLaMA2'
else
    echo "Your choose is not in MY candidations! Please check your Model name!"
fi
echo "Your choose ${MODEL_NAME}! Model Parameters should be initialized in the path \n ${MODEL_PATH}"


if [ ${Experiments_setting} = 'lora' ]
then
    DO_EVAL=True
    DO_TRAIN=True
    LORA=True
    LR=2e-4
    CHECKPOINT_DIR=None
    echo "Your choose ${Experiments_setting}! The experiment will be set as LORA model"
elif [ ${Experiments_setting} = 'all_parameters' ]
then
    DO_EVAL=True
    DO_TRAIN=True
    LORA=False
    LR=2e-5
    CHECKPOINT_DIR=None
    echo "Your choose ${Experiments_setting}! The experiment will be set as ALL_PARAMETERS model"
else
    echo "Your choose is not in MY candidations! Please CHECK your Experiments Setting!"
fi


SPEAKER_DATA_PATH=$(python ./code/data_process.py \
    --dataset ${dataset} \
    --historical_window ${historical_window} \
    --speaker_task True \
    --emotion_prediction False)
if [ $? -eq 0 ]; then
    echo "******************************************************************************************"
    echo "Data procession has executed successfully !"
    echo "******************************************************************************************"

else
    echo "Data procession script encountered an error."
fi

EMOTION_DATA_PATH=$(python ./code/data_process.py \
    --dataset ${dataset} \
    --historical_window ${historical_window} \
    --speaker_task None \
    --domain_base ${domain_base} \
    --emotion_prediction ${emotion_prediction})
if [ $? -eq 0 ]; then
    echo "******************************************************************************************"
    echo "Data procession has executed successfully !"
    echo "******************************************************************************************"

else
    echo "Data procession script encountered an error."
fi

DATA_SPEAKER_PATH=$(echo "$SPEAKER_DATA_PATH" | cut -d ',' -f 1)
DATA_WINDOW_PATH=$(echo "$EMOTION_DATA_PATH" | cut -d ',' -f 2)
Speaker_Model_output_dir=./experiments/${MODEL_NAME}/${Experiments_setting}/${dataset}/${speaker_task}_one
Content_Model_output_dir=./experiments/${MODEL_NAME}/${Experiments_setting}/${dataset}/${speaker_task}_two
echo "*********************************************"
echo "DATA_SPEAKER_PATH: $DATA_SPEAKER_PATH"
echo "*********************************************"
echo "DATA_WINDOW_PATH: $DATA_WINDOW_PATH"
echo "*********************************************"
echo "Speaker_Model_output_dir: $Speaker_Model_output_dir"
echo "*********************************************"
echo "Content_Model_output_dir: $Content_Model_output_dir"
echo "*********************************************"

# FLAG=0

if [ ${FLAG} -eq 1 ];
then
    echo "*********************************************"
    echo "Start to train on Speaker Identification task!"
    echo "*********************************************"
    deepspeed --master_port=29500 ./code/main_new.py \
        --dataset ${dataset} \
        --model_name_or_path ${MODEL_PATH} \
        --data_dir ${DATA_SPEAKER_PATH} \
        --output_dir ${Speaker_Model_output_dir} \
        --max_length ${MAX_LENGTH} \
        --batch_size ${BS} \
        --deepspeed_config ./code/data_utils/deepspeed_config.json \
        --gradient_accumulation_steps ${accumulations} \
        --eval_batch_size 8 \
        --num_train_epochs 3 \
        --save_steps 1000 \
        --lora ${LORA}\
        --learning_rate ${LR} \
        --do_train ${DO_TRAIN} \
        --do_eval ${DO_EVAL} \
        --statistic_mode False
        # --checkpoint_dir ${CHECKPOINT_DIR}

    echo "*******************************************************************"
    echo "Speaker Identification task has been achieved successfully!"
    echo "*******************************************************************"

    echo "*********************************************"
    echo "Start to train on Emotion Recognition task!"
    echo "*********************************************"
    deepspeed --master_port=29500 ./code/main_new.py \
        --dataset ${dataset} \
        --model_name_or_path ${MODEL_PATH} \
        --data_dir ${DATA_WINDOW_PATH} \
        --output_dir ${Content_Model_output_dir} \
        --max_length ${MAX_LENGTH} \
        --batch_size ${BS} \
        --deepspeed_config ./code/data_utils/deepspeed_config.json \
        --gradient_accumulation_steps ${accumulations} \
        --eval_batch_size 16 \
        --num_train_epochs 15 \
        --save_steps 1000 \
        --lora ${LORA}\
        --learning_rate ${LR} \
        --do_train ${DO_TRAIN} \
        --do_eval ${DO_EVAL} \
        --statistic_mode True \
        --checkpoint_dir ${Speaker_Model_output_dir}
fi
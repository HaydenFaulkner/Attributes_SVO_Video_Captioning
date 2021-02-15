# The default - a 512 single layer LSTM with attended features
python train.py --dataset msvd --captioner_type lstm --model_id lstm_1 --batch_size 8 --test_batch_size 8 --max_epochs 100
python train.py --dataset msrvtt --captioner_type lstm --model_id lstm_1  --batch_size 8 --test_batch_size 4 --max_epochs 200

# The default transformer - a 2048 single layer transformer with 8 heads
python train.py --dataset msvd --captioner_type transformer --model_id transformer_1 --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 2048  --captioner_heads 8
python train.py --dataset msrvtt --captioner_type transformer --model_id transformer_1  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 2048  --captioner_heads 8

# And the same but with 2 layers
python train.py --dataset msvd --captioner_type transformer --model_id transformer_2 --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 2048  --captioner_heads 8  --captioner_layers 2
python train.py --dataset msrvtt --captioner_type transformer --model_id transformer_2  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 2048  --captioner_heads 8  --captioner_layers 2

# Add visual encoders
python train.py --dataset msvd --captioner_type lstm --model_id lstm_1_ve --batch_size 8 --test_batch_size 8 --max_epochs 100 --input_encoder_layers 1
python train.py --dataset msrvtt --captioner_type lstm --model_id lstm_1_ve  --batch_size 8 --test_batch_size 4 --max_epochs 200 --input_encoder_layers 1

python train.py --dataset msvd --captioner_type transformer --model_id transformer_1_ve --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 2048  --captioner_heads 8 --input_encoder_layers 1
python train.py --dataset msrvtt --captioner_type transformer --model_id transformer_1_ve  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 2048  --captioner_heads 8 --input_encoder_layers 1

# Add grounding with single stage sigmoid with 5 top concepts (unconditioned, unordered)
python train.py --dataset msvd --grounder_type niuc --captioner_type lstm --model_id lstm_1_niuc --batch_size 8 --test_batch_size 8 --max_epochs 100 --num_concepts 5 --concepts_h5 sl_top_concepts
python train.py --dataset msrvtt --grounder_type niuc --captioner_type lstm --model_id lstm_1_niuc  --batch_size 8 --test_batch_size 4 --max_epochs 200 --num_concepts 5 --concepts_h5 sl_top_concepts

python train.py --dataset msvd --grounder_type niuc --captioner_type transformer --model_id transformer_1_niuc --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 2048  --captioner_heads 8 --num_concepts 5 --concepts_h5 sl_top_concepts
python train.py --dataset msrvtt --grounder_type niuc --captioner_type transformer --model_id transformer_1_niuc  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 2048  --captioner_heads 8 --num_concepts 5 --concepts_h5 sl_top_concepts

# with decoupling
python train.py --dataset msvd --grounder_type niuc --captioner_type lstm --model_id lstm_1_niuc_dec --batch_size 8 --test_batch_size 8 --max_epochs 100 --num_concepts 5 --concepts_h5 sl_top_concepts --decouple 1
python train.py --dataset msrvtt --grounder_type niuc --captioner_type lstm --model_id lstm_1_niuc_dec  --batch_size 8 --test_batch_size 4 --max_epochs 200 --num_concepts 5 --concepts_h5 sl_top_concepts --decouple 1

python train.py --dataset msvd --grounder_type niuc --captioner_type transformer --model_id transformer_1_niuc_dec --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 2048  --captioner_heads 8 --num_concepts 5 --concepts_h5 sl_top_concepts --decouple 1
python train.py --dataset msrvtt --grounder_type niuc --captioner_type transformer --model_id transformer_1_niuc_dec  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 2048  --captioner_heads 8 --num_concepts 5 --concepts_h5 sl_top_concepts --decouple 1

# Add grounding with multi stage (but not conditioned on past concepts) softmax with 5 top concepts (unconditioned, ordered)
python train.py --dataset msvd --grounder_type nioc --captioner_type lstm --model_id lstm_1_nioc --batch_size 8 --test_batch_size 8 --max_epochs 100 --num_concepts 5 --concepts_h5 sl_top_concepts
python train.py --dataset msrvtt --grounder_type nioc --captioner_type lstm --model_id lstm_1_nioc  --batch_size 8 --test_batch_size 4 --max_epochs 200 --num_concepts 5 --concepts_h5 sl_top_concepts

python train.py --dataset msvd --grounder_type nioc --captioner_type transformer --model_id transformer_1_nioc --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 2048  --captioner_heads 8 --num_concepts 5 --concepts_h5 sl_top_concepts
python train.py --dataset msrvtt --grounder_type nioc --captioner_type transformer --model_id transformer_1_nioc  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 2048  --captioner_heads 8 --num_concepts 5 --concepts_h5 sl_top_concepts

# Add grounding with iterative transformer with 5 top concepts (conditional, unordered)
python train.py --dataset msvd --grounder_type iuc --captioner_type lstm --model_id lstm_1_iuc --batch_size 8 --test_batch_size 8 --max_epochs 100 --grounder_size 2048 --grounder_heads 8 --num_concepts 5 --concepts_h5 sl_top_concepts
python train.py --dataset msrvtt --grounder_type iuc --captioner_type lstm --model_id lstm_1_iuc  --batch_size 8 --test_batch_size 4 --max_epochs 200 --grounder_size 2048 --grounder_heads 8 --num_concepts 5 --concepts_h5 sl_top_concepts

python train.py --dataset msvd --grounder_type iuc --captioner_type transformer --model_id transformer_1_iuc --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 2048 --captioner_heads 8 --grounder_size 2048 --grounder_heads 8 --num_concepts 5 --concepts_h5 sl_top_concepts
python train.py --dataset msrvtt --grounder_type iuc --captioner_type transformer --model_id transformer_1_iuc  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 2048 --captioner_heads 8 --grounder_size 2048 --grounder_heads 8 --num_concepts 5 --concepts_h5 sl_top_concepts

# Add grounding with iterative transformer with 5 top concepts (conditional, ordered)
python train.py --dataset msvd --grounder_type ioc --captioner_type lstm --model_id lstm_1_ioc --batch_size 8 --test_batch_size 8 --max_epochs 100 --grounder_size 2048 --grounder_heads 8 --num_concepts 5 --concepts_h5 sl_top_concepts
python train.py --dataset msrvtt --grounder_type ioc --captioner_type lstm --model_id lstm_1_ioc  --batch_size 8 --test_batch_size 4 --max_epochs 200 --grounder_size 2048 --grounder_heads 8 --num_concepts 5 --concepts_h5 sl_top_concepts

python train.py --dataset msvd --grounder_type ioc --captioner_type transformer --model_id transformer_1_ioc --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 2048 --captioner_heads 8 --grounder_size 2048 --grounder_heads 8 --num_concepts 5 --concepts_h5 sl_top_concepts
python train.py --dataset msrvtt --grounder_type ioc --captioner_type transformer --model_id transformer_1_ioc  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 2048 --captioner_heads 8 --grounder_size 2048 --grounder_heads 8 --num_concepts 5 --concepts_h5 sl_top_concepts

# Add grounding with iterative transformer and default svos ordered
python train.py --dataset msvd --grounder_type ioc --captioner_type lstm --model_id lstm_1_ioc_svo --batch_size 8 --test_batch_size 8 --max_epochs 100 --grounder_size 2048 --grounder_heads 8
python train.py --dataset msrvtt --grounder_type ioc --captioner_type lstm --model_id lstm_1_ioc_svo  --batch_size 8 --test_batch_size 4 --max_epochs 200 --grounder_size 2048 --grounder_heads 8

python train.py --dataset msvd --grounder_type ioc --captioner_type transformer --model_id transformer_1_ioc_svo --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 2048 --captioner_heads 8 --grounder_size 2048 --grounder_heads 8
python train.py --dataset msrvtt --grounder_type ioc --captioner_type transformer --model_id transformer_1_ioc_svo  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 2048 --captioner_heads 8 --grounder_size 2048 --grounder_heads 8

# Add grounding with multi stage (but not conditioned on past concepts) softmax with svos
python train.py --dataset msvd --grounder_type nioc --captioner_type lstm --model_id lstm_1_nioc_svo --batch_size 8 --test_batch_size 8 --max_epochs 100
python train.py --dataset msrvtt --grounder_type nioc --captioner_type lstm --model_id lstm_1_nioc_svo  --batch_size 8 --test_batch_size 4 --max_epochs 200

python train.py --dataset msvd --grounder_type nioc --captioner_type transformer --model_id transformer_1_nioc_svo --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 2048  --captioner_heads 8
python train.py --dataset msrvtt --grounder_type nioc --captioner_type transformer --model_id transformer_1_nioc_svo  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 2048  --captioner_heads 8

# with decoupling
python train.py --dataset msvd --grounder_type nioc --captioner_type lstm --model_id lstm_1_nioc_svo_dec --batch_size 8 --test_batch_size 8 --max_epochs 100 --decouple 1
python train.py --dataset msrvtt --grounder_type nioc --captioner_type lstm --model_id lstm_1_nioc_svo_dec  --batch_size 8 --test_batch_size 4 --max_epochs 200 --decouple 1

python train.py --dataset msvd --grounder_type nioc --captioner_type transformer --model_id transformer_1_nioc_svo_dec --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 2048  --captioner_heads 8 --decouple 1
python train.py --dataset msrvtt --grounder_type nioc --captioner_type transformer --model_id transformer_1_nioc_svo_dec  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 2048  --captioner_heads 8 --decouple 1

# Add grounding with iterative transformer and default svos ordered
python train.py --dataset msvd --grounder_type ioc --captioner_type lstm --model_id lstm_1_ioc_svo_dec --batch_size 8 --test_batch_size 8 --max_epochs 100 --grounder_size 2048 --grounder_heads 8 --decouple 1
python train.py --dataset msrvtt --grounder_type ioc --captioner_type lstm --model_id lstm_1_ioc_svo_dec  --batch_size 8 --test_batch_size 4 --max_epochs 200 --grounder_size 2048 --grounder_heads 8 --decouple 1

python train.py --dataset msvd --grounder_type ioc --captioner_type transformer --model_id transformer_1_ioc_svo_dec --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 2048 --captioner_heads 8 --grounder_size 2048 --grounder_heads 8 --decouple 1
python train.py --dataset msrvtt --grounder_type ioc --captioner_type transformer --model_id transformer_1_ioc_svo_dec  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 2048 --captioner_heads 8 --grounder_size 2048 --grounder_heads 8 --decouple 1


# Add grounding with iterative transformer and default svos ordered 2 cap layers
python train.py --dataset msvd --grounder_type ioc --captioner_type lstm --model_id lstm_1_ioc_svo_dec_28 --batch_size 8 --test_batch_size 8 --max_epochs 100 --grounder_size 2048 --grounder_layers 2 --grounder_heads 8 --decouple 1
python train.py --dataset msrvtt --grounder_type ioc --captioner_type lstm --model_id lstm_1_ioc_svo_dec_28  --batch_size 8 --test_batch_size 4 --max_epochs 200 --grounder_size 2048 --grounder_layers 2 --grounder_heads 8 --decouple 1

python train.py --dataset msvd --grounder_type ioc --captioner_type transformer --model_id transformer_1_ioc_svo_dec_28 --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 2048 --captioner_heads 8 --grounder_size 2048 --grounder_layers 2 --grounder_heads 8 --decouple 1
python train.py --dataset msrvtt --grounder_type ioc --captioner_type transformer --model_id transformer_1_ioc_svo_dec_28  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 2048 --captioner_heads 8 --grounder_size 2048 --grounder_layers 2 --grounder_heads 8 --decouple 1

# Add grounding with iterative transformer and default svos ordered 2 cap layers and encoder 1024
python train.py --dataset msvd --grounder_type ioc --captioner_type lstm --model_id lstm_1_ioc_svo_dec_28_enc_1024 --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 1024 --captioner_heads 8 --grounder_size 2048 --grounder_layers 2 --grounder_heads 8 --decouple 1 --input_encoder_layers 2 --input_encoder_heads 8 --input_encoder_size 2048 --input_encoding_size 1024 --att_size 1024
python train.py --dataset msrvtt --grounder_type ioc --captioner_type lstm --model_id lstm_1_ioc_svo_dec_28_enc_1024  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 1024 --captioner_heads 8 --grounder_size 2048 --grounder_layers 2 --grounder_heads 8 --decouple 1 --input_encoder_layers 2 --input_encoder_heads 8 --input_encoder_size 2048 --input_encoding_size 1024
#
python train.py --dataset msvd --grounder_type ioc --captioner_type transformer --model_id transformer_1_ioc_svo_dec_28_enc_1024 --batch_size 8 --test_batch_size 8 --max_epochs 100 --captioner_size 2048 --captioner_heads 8 --grounder_size 2048 --grounder_layers 2 --grounder_heads 8 --decouple 1 --input_encoder_layers 2 --input_encoder_heads 8 --input_encoder_size 2048 --input_encoding_size 1024
python train.py --dataset msrvtt --grounder_type ioc --captioner_type transformer --model_id transformer_1_ioc_svo_dec_28_enc_1024  --batch_size 8 --test_batch_size 4 --max_epochs 200 --captioner_size 2048 --captioner_heads 8 --grounder_size 2048 --grounder_layers 2 --grounder_heads 8 --decouple 1 --input_encoder_layers 2 --input_encoder_heads 8 --input_encoder_size 2048 --input_encoding_size 1024

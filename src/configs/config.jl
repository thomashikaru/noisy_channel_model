using ArgParse

function parse_commandline(args = ARGS)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--base_directory"
        arg_type = String
        default = ".."
        "--mode"
        help = "e.g. debug_one"
        arg_type = String
        default = "debug_one"
        "--debug_input"
        arg_type = String
        default = "the boy licked the ball ."
        "--test_sents_file"
        help = "file containing test sentences"
        arg_type = String
        default = "data/gibson2013/dopo_to/sentences.txt"
        "--train_sents_file"
        help = "file containing train sentences (for ngram)"
        arg_type = String
        "--sent_id"
        help = "sentence num"
        arg_type = Int
        default = 1
        "--output_dir"
        help = "directory to save results in"
        arg_type = String
        default = "log/test_one"
        "--num_particles"
        help = "number of particles"
        arg_type = Int
        default = 32
        "--ipa"
        help = "set to use IPA edit distance, otherwise will use orthographic edit distance"
        arg_type = Bool
        default = false
        "--conditional_rejuv"
        help = "whether to do conditional rejuvenation"
        arg_type = Bool
        default = true
        "--logprob_thresh"
        help = "inflection point for logprob"
        arg_type = Float64
        default = -10
        "--logprob_spread"
        help = "spread of sigmoid: set larger values for wider spread"
        arg_type = Float64
        default = 0.5
        "--lookback"
        help = "lookback in words"
        arg_type = Int
        default = 1
        "--second_pass_rejuv"
        help = "whether to do second-pass rejuvenation"
        arg_type = Bool
        default = true
        "--second_pass_rejuv_p"
        help = "probability of a particle performing second-pass rejuv"
        arg_type = Float64
        default = 1.0
        "--num_samples"
        help = "number of posterior samples to draw"
        arg_type = Int
        default = 100
        "--lm_method"
        help = "language model"
        arg_type = String
        default = "gpt2"
        "--num_threads"
        help = "number of threads"
        arg_type = Int
        default = 1
        "--normal_alpha"
        help = "normal alpha parameter"
        arg_type = Int
        default = 10
        "--error_alpha"
        help = "error alpha parameter"
        arg_type = Int
        default = 1
        "--vocab_file"
        arg_type = String
        default = "../data/gibson2013/dopo_to/vocab_augmented.txt"
        "--vocab_ipa_file"
        arg_type = String
        default = "../data/gibson2013/dopo_to/vocab_ipa_augmented.txt"
        "--actions"
        arg_type = String
        default = "normal,sem_sub,form_sub,insert,skip,morph_sub"
    end

    return parse_args(args, s)
end

# args = parse_commandline(String[])
args = parse_commandline()

println("Arguments:")
println(args)

const BASE_DIRECTORY = args["base_directory"]
const LM_METHOD = args["lm_method"]

const WORD_FREQ_FILE = "$BASE_DIRECTORY/data/SUBTLEXus74286wordstextversion.txt"

# resampling threshold
const ESS_THRESH = Inf # always resample

# whether to print debugging statements
const VERBOSE = false

# Provide the path to your GloVe vectors file, only contains the top 10k most common words
const GLOVE_VECTOR_FILE = "$BASE_DIRECTORY/data/glove_vectors.txt"

const DISFL_LIST = ["um", "uh", "ah", "eh", "er", "ew", "hm", "mm", "uhm"]
const DISFL_SET = Set(DISFL_LIST)

const ACTION_LIST = map(String, split(args["actions"], ","))

# ALPHAS for Dirichlet Prior
normal_alpha = args["normal_alpha"]
error_alpha = args["error_alpha"]
alphas = [error_alpha for _ = 1:length(ACTION_LIST)-1]
pushfirst!(alphas, normal_alpha)

const EOS_TOKENS = [".", "?", "!"]

const REJUVENATE_ORDER = "SHUFFLE"

const DO_TYPO = true

const UTTERANCES = [
    "the boy handed the pencil the girl .",
    "the mother gave the candle the daughter .",
]


GPT_PROMPT = String[]

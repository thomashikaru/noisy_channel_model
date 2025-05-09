include("./configs/config.jl")

import Gen
import Gen: @gen, @dist
using StatsBase
using Memoize
using NNlib
using Random
using CSV, DataFrames

# lookup of irregular english singular/plurals
const MORPH_SUBS = Dict([
    ("has", "have"),
    ("have", "has"),
    ("was", "were"),
    ("were", "was"),
    ("is", "are"),
    ("are", "is"),
    ("company", "companies"),
    ("companies", "company"),
    ("discovery", "discoveries"),
    ("discoveries", "discovery"),
    ("family", "families"),
    ("families", "family"),
    ("shelf", "shelves"),
    ("shelves", "shelf"),
    ("beach", "beaches"),
    ("beaches", "beach"),
    ("box", "boxes"),
    ("boxes", "box"),
    ("amenity", "amenities"),
    ("cavalry", "cavalries"),
    ("cavity", "cavities"),
    ("phonology", "phonologies"),
    ("phrenology", "phrenologies"),
    ("humidity", "humidities"),
    ("hypothesis", "hypotheses"),
    ("immorality", "immoralities"),
    ("immortality", "immortalities"),
    ("peasantry", "peasantries"),
    ("pleasantry", "pleasantries"),
    ("tapestry", "tapestries"),
    ("travesty", "travesties"),
    ("nobility", "nobilities"),
    ("mobility", "mobilities"),
    ("immunity", "immunities"),
    ("alumnus", "alumni"),
    ("radius", "radii"),
    ("balcony", "balconies"),
    ("bounty", "bounties"),
    ("mercenary", "mercenaries"),
    ("missionary", "missionaries"),
    ("density", "densities"),
    ("destiny", "destinies"),
    ("dynasty", "dynasties"),
    ("fallacy", "fallacies"),
])

OTHER_PUNCT_TOKENS = [","]

# load vocabulary from file
vocab_list = split(rstrip(read(args["vocab_file"], String)), "\n")
DISFL_LIST_EXTRA = [x for x in DISFL_LIST if !(x in vocab_list)]
vocab_list = [vocab_list; DISFL_LIST_EXTRA]
vocab_set = Set(vocab_list)

# function to get index of a given word in the vocabulary list
@memoize function get_vocab_idx(word::AbstractString)::Int
    if !(word in vocab_list)
        println("$(word) not in vocab list")
    end
    return findfirst(x -> x == word, vocab_list)
end

# IPA conversion of vocab list
vocab_ipa_list = split(rstrip(read(args["vocab_ipa_file"], String)), "\n")
vocab_ipa_list = [vocab_ipa_list; DISFL_LIST_EXTRA]

# Read the CSV file into a DataFrame
df = CSV.read(WORD_FREQ_FILE, DataFrame; delim = '\t')

# Create a dictionary mapping each word to its frequency
word2freq = Dict(row.Word => row.Lg10WF for row in eachrow(df))

# Get the minimum frequency from the CSV data
min_freq = minimum(df.Lg10WF)

# Given your vocab list, look up each word's frequency; if not found, assign min_freq
unigram_freq = [get(word2freq, word, min_freq) for word in vocab_list]
unigram_freq = exp10.(unigram_freq)
unigram_probs = unigram_freq / sum(unigram_freq)

# word frequencies (excluding EOS tokens)
# we need this because there are cases where we want to sample from the unigram probs
# without allowing a EOS token
unigram_probs_no_eos = copy(unigram_probs)
for x in EOS_TOKENS
    unigram_probs_no_eos[get_vocab_idx(x)] = 1e-20
end
unigram_probs_no_eos = unigram_probs_no_eos / sum(unigram_probs_no_eos)

function normalize_array(a::Vector{Float64})::Vector{Float64}
    return a / sum(a)
end

# get a probability distribution over vocabulary items, proportional to semantic similarity
@memoize function get_sem_sub_ps(word::AbstractString, param::Float64)::Vector{Float64}
    if !haskey(glove_lookup, word)
        return unigram_probs_no_eos
    end
    e1 = glove_lookup[word]
    sims = map(
        x ->
            x == word ? 0 : (haskey(glove_lookup, x) ? cosine_sim(e1, glove_lookup[x]) : 0),
        vocab_list,
    )

    # using exponentiation rather than softmax
    minval = minimum(sims) - 0.01
    new_sims = fill(0.0, length(sims))

    for idx in partialsortperm(sims, 1:20; rev = true)
        new_sims[idx] = ((sims[idx] - minval) / (1 - minval))^param
    end

    for x in EOS_TOKENS
        new_sims[get_vocab_idx(x)] = 0
    end

    return normalize_array(new_sims)
end

# finds words which can be morphological neighbors of the input word
@memoize function get_morph_sub_ps(word::AbstractString)::Vector{Float64}
    if haskey(MORPH_SUBS, word)
        return word_onehot(MORPH_SUBS[word])
    end
    if word[end] == 's'
        if word[1:end-1] in vocab_set
            return word_onehot(word[1:end-1])
        end
    else
        if word * "s" in vocab_set
            return word_onehot(word * "s")
        end
    end
    return fill(0.0, length(vocab_list))
end

# get a probability distribution over vocabulary items, proportional to form-based similarity
@memoize function get_phon_sub_ps(word::AbstractString, param::Float64)::Vector{Float64}
    if word == "<nonword>"
        return unigram_probs_no_eos
    end

    word_form =
        args["ipa"] ? vocab_ipa_list[get_vocab_idx(word)] : vocab_list[get_vocab_idx(word)]

    wl = length(word_form)
    distances = map(
        x -> x == word_form ? wl : Levenshtein()(word_form, x),
        args["ipa"] ? vocab_ipa_list : vocab_list,
    )
    sims = map(x -> x >= 5 ? 0.0 : param^x, distances)

    for x in EOS_TOKENS
        sims[get_vocab_idx(x)] = 0
    end

    # it's possible that the word has no similar neighbors
    if sum(sims) == 0
        return unigram_probs_no_eos
    end

    return normalize_array(sims)
end

# cosine similarity between two vectors
function cosine_sim(v1::Vector{Float64}, v2::Vector{Float64})::Float64
    return 1.0 - cosine_dist(v1, v2)
end

# return a one-hot vector between the vocab list and a given word
function word_onehot(word::AbstractString)::Vector{Float64}
    return Float64.(vocab_list .== word)
end


ENV["PYTHON"] = "/usr/local/bin/python3.11"
using PyCall

sys = pyimport("sys")
pycall(
    sys["path"].append,
    PyObject,
    "hfppl_custom",
)

println("Importing transformers...")
transformers = pyimport("transformers")

hfppl = pyimport("hfppl")
torch = pyimport("torch")
np = pyimport("numpy")

println("Loading model...")
model_name = LM_METHOD
base_model_name = LM_METHOD

gpt_model = hfppl.CachedCausalLM.from_pretrained(
    base_model_name,
    auth_token = false,
    load_in_8bit = false,
)
tokenizer = gpt_model.tokenizer

function tokenize_custom(word::AbstractString, prepend_space::Bool)::Vector{Int}
    if prepend_space
        return tokenizer.encode(" $(word)", add_special_tokens = false)
    else
        return tokenizer.encode(word, add_special_tokens = false)
    end
end

vocab_list_tokenized_bos = []
for word in vocab_list
    push!(
        vocab_list_tokenized_bos,
        tokenize_custom(uppercasefirst(word), !(word ∈ EOS_TOKENS || word ∈ OTHER_PUNCT_TOKENS))
    )
end

vocab_list_tokenized = []
for word in vocab_list
    push!(
        vocab_list_tokenized,
        tokenize_custom(word, !(word ∈ EOS_TOKENS || word ∈ OTHER_PUNCT_TOKENS))
    )
end

first_tokens_bos = [item[1] for item in vocab_list_tokenized_bos]
vocab_map_bos =
    Dict(x => count(y -> y == x, first_tokens_bos) for x in first_tokens_bos)
token_share_count_bos = [vocab_map_bos[x] for x in first_tokens_bos]

first_tokens = [item[1] for item in vocab_list_tokenized]
vocab_map = Dict(x => count(y -> y == x, first_tokens) for x in first_tokens)
token_share_count = [vocab_map[x] for x in first_tokens]

eos_tokenized = []
for item in EOS_TOKENS
    tokenized = tokenizer.encode(item, add_special_tokens = false)
    @assert length(tokenized) == 1
    push!(eos_tokenized, tokenized[1])
end

other_punct_tokenized = []
for item in OTHER_PUNCT_TOKENS
    tokenized = tokenizer.encode(item, add_special_tokens = false)
    @assert length(tokenized) == 1
    push!(other_punct_tokenized, tokenized[1])
end

struct GPTWordDist <: Gen.Distribution{String}
    model::Any
    tokenizer::Any
end

(dist::GPTWordDist)(prefix::Vector{<:AbstractString}) = Gen.random(dist, prefix)

function Gen.random(dist::GPTWordDist, prefix::Vector{<:AbstractString})

    if length(prefix) > 0
        input_ids =
            vcat(dist.tokenizer.bos_token_id, vcat([tokenize_custom(x, !(x ∈ OTHER_PUNCT_TOKENS || x ∈ EOS_TOKENS)) for x in prefix]...))
    else
        input_ids = [dist.tokenizer.bos_token_id]
    end

    generated = ""
    voc_toks = length(prefix) == 0 ? vocab_list_tokenized_bos : vocab_list_tokenized
    possible_completions = copy(voc_toks)

    count = 1
    while true

        if count > 15
            break
        end

        # get logits for current prefix
        logits = dist.model.next_token_logprobs_unbatched(input_ids)

        # empty set
        valid_next_tokens = Set()

        # allow generating EOS punctuation as stand-alone tokens only (not as part of a larger word)
        if count == 1 && length(prefix) > 0
            valid_next_tokens = valid_next_tokens ∪ Set(eos_tokenized) ∪ Set(other_punct_tokenized)
        end

        # Get the set of valid next tokens based on current completions
        for remaining_tokens in possible_completions
            if length(remaining_tokens) > 0
                push!(valid_next_tokens, remaining_tokens[1])
            end
        end

        # we want to allow generation of a new word if a valid vocab word has been formed
        # (but a new word will terminate the loop)
        if (generated in vocab_set) ||
           (length(prefix) == 0 && lowercasefirst(generated) in vocab_set)
            valid_next_tokens =
                valid_next_tokens ∪ Set(first_tokens) ∪ Set(eos_tokenized) ∪ Set(other_punct_tokenized)
        end

        # Mask all tokens not in the valid next token set
        mask = fill(-Inf, length(logits))
        valid_next_tokens_vector = collect(valid_next_tokens)

        # change the indexing 0-index (python/huggingface) -> 1-index (julia)
        valid_next_tokens_vector = valid_next_tokens_vector .+ 1

        # this should never happen
        @assert length(valid_next_tokens_vector) > 0 "No valid next tokens: generated=$(generated) input_ids=$(input_ids) count=$(count) tokens=$(tokenizer.convert_ids_to_tokens(input_ids))"

        # copy the valid logits
        mask[valid_next_tokens_vector] = logits[valid_next_tokens_vector]

        # apply softmax
        probs = NNlib.softmax(mask, dims = 1)

        # sample a token
        new_tok_id = Gen.categorical(probs) - 1
        new_tok = dist.tokenizer.convert_ids_to_tokens([new_tok_id])[1]

        # if a new word character is generated, break loop
        if length(generated) > 0 && (new_tok[1] == 'Ġ' || new_tok ∈ EOS_TOKENS || new_tok ∈ OTHER_PUNCT_TOKENS)
            break
        end

        # Filter possible completions to retain only those starting with the generated token
        new_possible_completions = []
        for remaining_tokens in possible_completions
            if length(remaining_tokens) > 1 && remaining_tokens[1] == new_tok_id
                # Move to the next token in this word's sequence
                push!(new_possible_completions, remaining_tokens[2:end])
            end
        end
        possible_completions = new_possible_completions

        # update the input_ids and generated string
        push!(input_ids, new_tok_id)

        # if the word is not sentence initial, it will start with special Ġ character
        if count == 1 && !(new_tok ∈ EOS_TOKENS || new_tok ∈ OTHER_PUNCT_TOKENS)
            generated *= new_tok[nextind(new_tok, 1):end]
        else
            generated *= new_tok
        end
        count += 1
    end

    @assert generated in vocab_set || lowercasefirst(generated) in vocab_set "$(generated) not in vocabulary"

    if !(generated ∈ vocab_set)
        return lowercasefirst(generated)
    else
        return generated
    end
end

function Gen.logpdf(dist::GPTWordDist, word::AbstractString, prefix::Vector{<:AbstractString})

    logprobs = 0.0

    if length(prefix) > 0
        input_ids =
            vcat(dist.tokenizer.bos_token_id, vcat([tokenize_custom(x, !(x ∈ OTHER_PUNCT_TOKENS || x ∈ EOS_TOKENS)) for x in prefix]...))
    else
        input_ids = [dist.tokenizer.bos_token_id]
    end
    word_ids = tokenize_custom(
        length(prefix) == 0 ? uppercasefirst(word) : word,
        !(word ∈ EOS_TOKENS || word ∈ OTHER_PUNCT_TOKENS),
    )
    generated = ""

    voc_toks = length(prefix) == 0 ? vocab_list_tokenized_bos : vocab_list_tokenized
    possible_completions = copy(voc_toks)

    for (count, word_id) in enumerate(word_ids)

        # ...
        new_tok = dist.tokenizer.convert_ids_to_tokens([word_id])[1]

        # get logits for current prefix
        logits = dist.model.next_token_logprobs_unbatched(input_ids)

        # empty set
        valid_next_tokens = Set()

        # allow generating EOS punctuation as stand-alone tokens only (not as part of a larger word)
        if count == 1 && length(prefix) > 0
            valid_next_tokens = valid_next_tokens ∪ Set(eos_tokenized) ∪ Set(other_punct_tokenized)
        end

        # Get the set of valid next tokens based on current completions
        for remaining_tokens in possible_completions
            if length(remaining_tokens) > 0
                push!(valid_next_tokens, remaining_tokens[1])
            end
        end

        # we want to allow generation of a new word if a valid vocab word has been formed
        # (but a new word will terminate the loop)
        if (generated in vocab_set) ||
           (length(prefix) == 0 && lowercasefirst(generated) in vocab_set)
            valid_next_tokens =
                valid_next_tokens ∪ Set(first_tokens) ∪ Set(eos_tokenized) ∪ Set(other_punct_tokenized)
        end

        # Mask all tokens not in the valid next token set
        mask = fill(-Inf, length(logits))
        valid_next_tokens_vector = collect(valid_next_tokens)

        # change the indexing 0-index (python/huggingface) -> 1-index (julia)
        valid_next_tokens_vector = valid_next_tokens_vector .+ 1

        # this should never happen?
        length(valid_next_tokens_vector) == 0 && break
        if length(valid_next_tokens) == 0
            return -Inf
        end
        @assert length(valid_next_tokens_vector) > 0 "No valid next tokens: input_ids=$(input_ids) count=$(count) tokens=$(tokenizer.convert_ids_to_tokens(input_ids))"

        # copy the valid logits
        mask[valid_next_tokens_vector] = logits[valid_next_tokens_vector]

        # apply softmax
        probs = NNlib.softmax(mask, dims = 1)
        # probs = NNlib.softmax(logits, dims = 1)

        # update the log probability
        logprobs += log(probs[word_id+1])

        # Filter possible completions to retain only those starting with the generated token
        new_possible_completions = []
        for remaining_tokens in possible_completions
            if length(remaining_tokens) > 1 && remaining_tokens[1] == word_id
                # Move to the next token in this word's sequence
                push!(new_possible_completions, remaining_tokens[2:end])
            end
        end
        possible_completions = new_possible_completions

        # update the input_ids and generated string
        push!(input_ids, word_id)

        # if the word is not sentence initial, it will start with special Ġ character
        if count == 1 && !(new_tok ∈ EOS_TOKENS || new_tok ∈ OTHER_PUNCT_TOKENS)
            generated *= new_tok[nextind(new_tok, 1):end]
        else
            generated *= new_tok
        end
    end

    return logprobs
end

Gen.logpdf_grad(dist::GPTWordDist, word::String, prefix::Vector{<:AbstractString}) = (nothing, nothing)
Gen.has_output_grad(dist::GPTWordDist) = false
Gen.has_argument_grads(dist::GPTWordDist) = (false,)
Gen.is_discrete(dist::GPTWordDist) = true

gpt_word_dist = GPTWordDist(gpt_model, tokenizer)
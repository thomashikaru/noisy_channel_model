using Distributions, KernelDensity
using Distances, StringDistances
using DelimitedFiles

include("dirichlet.jl")
include("gen_lm.jl")

function read_glove_vectors(file_path)
    # Read the GloVe vectors from the text file
    return readdlm(file_path, ' ', Any, quotes = false)
end

# Call the function to read the vectors
glove = read_glove_vectors(GLOVE_VECTOR_FILE)
words = string.(glove[:, 1])
glove_vecs = Float64.(glove[:, 2:end])
glove_lookup = Dict{String,Vector{Float64}}()

# Populate the dictionary with strings associated with matrix rows
for (i, w) in enumerate(words)
    glove_lookup[w] = glove_vecs[i, :]
end

# custom distributions
@dist action_dist(probs) = ACTION_LIST[Gen.categorical(probs)]
@dist word_dist(probs) = vocab_list[Gen.categorical(probs)]
@dist nonword_dist() = ["<nonword>"][Gen.categorical([1])]
@dist disfl_dist() = DISFL_LIST[Gen.categorical(fill(1 / length(DISFL_LIST), length(DISFL_LIST)))]
@dist disfl_dist_one(probs) = DISFL_LIST[Gen.categorical(probs)]
@dist dummy_dist(a::Int) = a + [0][Gen.categorical([1])]
@dist lookahead_dist(probs) = [0, 1][Gen.categorical(probs)]
@dist backtrack_dist(i) = 1 + Gen.categorical(fill(1 / (i - 1), i - 1))

# mixture distribution of two word dists
word_mixture_dist = Gen.HeterogeneousMixture([word_dist, word_dist])

# convert the name (String) of an action to a 1-hot vector compatible with action_dist
@memoize function action_onehot(action::String)
    return map(x -> x == action ? 1 : 0, ACTION_LIST)
end

# a Language Model generation state
struct LMState
    prompt::Vector{<:AbstractString}
    context::Vector{<:AbstractString}
    oblig::Vector{<:AbstractString}
    flag::Bool
end

# kernel that is applied to a LMState to produce the next state in sentence generation: gpt version
@gen function lm_kernel(t::Int, prev_state::LMState)
    if prev_state.flag
        word = {:w} ~ nonword_dist()
        return LMState(
            copy(prev_state.prompt),
            copy(prev_state.context),
            copy(prev_state.oblig),
            true,
        )
    end
    if length(prev_state.oblig) > 0
        @assert prev_state.oblig[1] in vocab_set "$(prev_state.oblig[1]) not in vocab_set"
        word = {:w} ~ word_dist(word_onehot(prev_state.oblig[1]))
        newflag = word ∈ EOS_TOKENS || prev_state.flag
        newoblig = copy(prev_state.oblig)
        popfirst!(newoblig)
        newcontext = copy(prev_state.context)
        push!(newcontext, word)
        return LMState(copy(prev_state.prompt), newcontext, newoblig, newflag)
    else
        word = {:w} ~ gpt_word_dist(vcat(prev_state.prompt, prev_state.context))
        newflag = word ∈ EOS_TOKENS || prev_state.flag
        newcontext = copy(prev_state.context)
        push!(newcontext, word)
        return LMState(copy(prev_state.prompt), newcontext, String[], newflag)
    end
end
generate_intended_sentence = Gen.Unfold(lm_kernel)

# a model state
struct ModelState
    action_prior::Vector{Float64}
    form_sub_param::Float64
    sem_sub_param::Float64
    sent::Vector{<:AbstractString}
    i::Int
end

@gen function model_kernel(t::Int, prev_state::ModelState)
    action = {:action} ~ action_dist(prev_state.action_prior)

    if prev_state.i > length(prev_state.sent) &&
       !(action in ["disfl", "insert", "backtrack"])
        word = {:word} ~ nonword_dist()
        i = {:idx} ~ dummy_dist(prev_state.i)
    elseif prev_state.i > length(prev_state.sent) + 1
        word = {:word} ~ nonword_dist()
        i = {:idx} ~ dummy_dist(prev_state.i)
    elseif action == "normal"
        word = {:word} ~ word_dist(word_onehot(prev_state.sent[prev_state.i]))
        i = {:idx} ~ dummy_dist(prev_state.i + 1)
    elseif action == "sem_sub"
        word =
            {:word} ~ word_dist(
                normalize_array(
                    get_sem_sub_ps(prev_state.sent[prev_state.i], prev_state.sem_sub_param),
                ),
            )
        i = {:idx} ~ dummy_dist(prev_state.i + 1)
    elseif action == "form_sub"
        word =
            {:word} ~ word_dist(
                normalize_array(
                    get_form_sub_ps(
                        prev_state.sent[prev_state.i],
                        prev_state.form_sub_param,
                    ),
                ),
            )
        i = {:idx} ~ dummy_dist(prev_state.i + 1)
    elseif action == "morph_sub"
        morph_subs = get_morph_sub_ps(prev_state.sent[prev_state.i])
        if sum(morph_subs) == 0
            word = {:word} ~ nonword_dist()
        else
            word = {:word} ~ word_dist(morph_subs)
        end
        i = {:idx} ~ dummy_dist(prev_state.i + 1)
    elseif action == "disfl"
        word = {:word} ~ disfl_dist()
        i = {:idx} ~ dummy_dist(prev_state.i)
    elseif action == "insert"
        word = {:word} ~ word_dist(unigram_probs)
        i = {:idx} ~ dummy_dist(prev_state.i)
    elseif action == "backtrack"
        i = {:idx} ~ backtrack_dist(prev_state.i)
        if prev_state.i == 1
            word = {:word} ~ nonword_dist()
        else
            word = {:word} ~ word_dist(word_onehot(prev_state.sent[max(1, i - 1)]))
        end
    elseif action == "skip"
        if prev_state.i == length(prev_state.sent)
            i = {:idx} ~ dummy_dist(prev_state.i)
            word = {:word} ~ nonword_dist()
        else
            i = {:idx} ~ dummy_dist(prev_state.i + 2)
            word =
                {:word} ~ word_dist(
                    word_onehot(
                        prev_state.sent[min(max(1, i - 1), length(prev_state.sent))],
                    ),
                )
        end
    end
    return ModelState(
        copy(prev_state.action_prior),
        prev_state.form_sub_param,
        prev_state.sem_sub_param,
        copy(prev_state.sent),
        i,
    )
end
generate_noisy_sentence = Gen.Unfold(model_kernel)

# model to generate a single noisy sentence
@gen function model_unfold(T::Int)

    # get a probability distribution over actions
    action_prior = {:action_prior} ~ Main.dirichlet(alphas)

    # sample substitution parameters
    form_sub_param = {:form_sub_param} ~ Gen.beta(2, 11) # mode = 0.1 (param^x)
    sem_sub_param = {:sem_sub_param} ~ Gen.gamma(6, 1) # mode = 5 (x^param)

    # i represents the current index within the current sentence
    i = 1

    # lookahead is needed to make skips possible:
    # (there is some probability that future words are generated)
    lazy_prob = {:lazy_prob} ~ Gen.beta(5, 5)
    lookahead = {:lookahead} ~ lookahead_dist([lazy_prob, 1 - lazy_prob])

    intended_sent =
        {:intended_sent} ~ generate_intended_sentence(
            T + lookahead,
            LMState(GPT_PROMPT, String[], String[], false),
        )

    sent = length(intended_sent) == 0 ? String[] : intended_sent[end].context

    init_state = ModelState(action_prior, form_sub_param, sem_sub_param, sent, i)
    noisy_sent = {:noisy_sent} ~ generate_noisy_sentence(T, init_state)

    result = (action_prior, join(sent, " "))
    return result
end;

struct ModelProposalState
    form_sub_param::Float64
    sem_sub_param::Float64
    sent::Vector{<:AbstractString}
    obss::Vector{<:AbstractString}
    action::String
    i::Int
    tt::Int
    add::Bool
end

@gen function model_proposal_kernel(t::Int, state::ModelProposalState)

    if t == state.tt && state.add && state.action == "normal"
        {:action} ~ action_dist(action_onehot("skip"))
        new_i = state.i + 2
        {:idx} ~ dummy_dist(new_i)
        {:word} ~ word_dist(word_onehot(state.obss[t]))
    elseif t == state.tt && state.add && state.action == "insert"
        {:action} ~ action_dist(action_onehot("normal"))
        new_i = state.i + 1
        {:idx} ~ dummy_dist(new_i)
        {:word} ~ word_dist(word_onehot(state.obss[t]))
    elseif t == state.tt && !state.add && state.action == "normal"
        {:action} ~ action_dist(action_onehot("insert"))
        new_i = state.i
        {:idx} ~ dummy_dist(new_i)
        {:word} ~ word_dist(word_onehot(state.obss[t]))
    elseif t == state.tt && !state.add && state.action == "skip"
        {:action} ~ action_dist(action_onehot("normal"))
        new_i = state.i + 1
        {:idx} ~ dummy_dist(new_i)
        {:word} ~ word_dist(word_onehot(state.obss[t]))
    elseif "disfl" in ACTION_LIST && state.obss[t] in DISFL_LIST
        {:action} ~ action_dist(action_onehot("disfl"))
        new_i = state.i
        {:idx} ~ dummy_dist(new_i)
        {:word} ~ disfl_dist_one(map(x -> x == state.obss[t] ? 1 : 0, DISFL_LIST))
    elseif "normal" in ACTION_LIST && length(state.sent) >= state.i && state.obss[t] == state.sent[state.i]
        {:action} ~ action_dist(action_onehot("normal"))
        new_i = state.i + 1
        {:idx} ~ dummy_dist(new_i)
        {:word} ~ word_dist(word_onehot(state.obss[t]))
    elseif "insert" in ACTION_LIST &&
           length(state.obss) >= t + 1 &&
           length(state.sent) >= state.i &&
           state.obss[t+1] == state.sent[state.i]
        {:action} ~ action_dist(action_onehot("insert"))
        new_i = state.i
        {:idx} ~ dummy_dist(new_i)
        {:word} ~ word_dist(word_onehot(state.obss[t]))
    elseif "skip" in ACTION_LIST &&
           length(state.sent) >= state.i + 1 &&
           state.obss[t] == state.sent[state.i+1] &&
           Gen.bernoulli(0.9)
        {:action} ~ action_dist(action_onehot("skip"))
        new_i = state.i + 2
        {:idx} ~ dummy_dist(new_i)
        {:word} ~ word_dist(word_onehot(state.obss[t]))
    elseif "backtrack" in ACTION_LIST &&
           length(state.sent) >= state.i - 1 &&
           state.obss[t] in state.sent[1:state.i-1] &&
           Gen.bernoulli(0.5)
        {:action} ~ action_dist(action_onehot("backtrack"))
        new_i = findfirst(x -> x == state.obss[t], state.sent) + 1
        {:idx} ~ dummy_dist(new_i)
        {:word} ~ word_dist(word_onehot(state.obss[t]))
    elseif "morph_sub" in ACTION_LIST &&
           length(state.sent) >= state.i &&
           get_morph_sub_ps(state.sent[state.i]) == state.obss[t]
        {:action} ~ action_dist(action_onehot("morph_sub"))
        new_i = state.i + 1
        {:idx} ~ dummy_dist(new_i)
    elseif length(state.sent) >= state.i
        if "sem_sub" in ACTION_LIST &&
           (
               get_sem_sub_ps(state.sent[state.i], state.sem_sub_param)[get_vocab_idx(
                   state.obss[t],
               )] != 0
           ) &&
           Gen.bernoulli(0.5)
            {:action} ~ action_dist(action_onehot("sem_sub"))
            new_i = state.i + 1
            {:idx} ~ dummy_dist(new_i)
            {:word} ~ word_dist(word_onehot(state.obss[t]))
        elseif "form_sub" in ACTION_LIST && (
            get_form_sub_ps(state.sent[state.i], state.form_sub_param)[get_vocab_idx(
                state.obss[t],
            )] != 0)
            {:action} ~ action_dist(action_onehot("form_sub"))
            new_i = state.i + 1
            {:idx} ~ dummy_dist(new_i)
            {:word} ~ word_dist(word_onehot(state.obss[t]))
        else
            {:action} ~ action_dist(action_onehot("insert"))
            new_i = state.i
            {:idx} ~ dummy_dist(new_i)
            {:word} ~ word_dist(word_onehot(state.obss[t]))
        end
    elseif "insert" in ACTION_LIST
        {:action} ~ action_dist(action_onehot("insert"))
        new_i = state.i
        {:idx} ~ dummy_dist(new_i)
        {:word} ~ word_dist(word_onehot(state.obss[t]))
    end

    return ModelProposalState(
        state.form_sub_param,
        state.sem_sub_param,
        state.sent,
        state.obss,
        state.action,
        new_i,
        state.tt,
        state.add
    )
end
generate_noisy_sent_proposal_unfold = Gen.Unfold(model_proposal_kernel)
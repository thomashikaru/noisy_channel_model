include("gen_model.jl")

using Statistics
using SortingAlgorithms
using LinearAlgebra
using Base.Threads
using Printf

ENV["JULIA_NUM_THREADS"] = args["num_threads"]

function custom_sigmoid(x::Float64, center::Float64, spread::Float64)
    num = exp(spread * (x - center))
    return num / (1 + num)
end

function mean(x::Vector{Float64})::Float64
    return sum(x) / length(x)
end

function top_x_mask(arr::AbstractVector, x::Int)
    threshold = partialsort(arr, length(arr) - x + 1)[1]
    result = [val ≥ threshold ? val : 0 for val in arr]
    return result / sum(result)
end

function log_rejuv_result!(local_results, t, particle, move, accepted, t_prime)
    push!(
        local_results[threadid()],
        Dict(
            "t" => t,
            "particle" => particle,
            "move" => move,
            "accepted" => accepted,
            "t_prime" => t_prime,
        ),
    )
end

# takes an array of traces and saves the relevant information to files
function analyze_traces(traces, n, output_dir)
    step = []
    inferred_words = []
    inferred_actions = []
    inferred_idxs = []
    obs_words = []
    for tr in traces
        sent = tr[:intended_sent]
        sent = sent[end].context
        for t = 1:n
            push!(step, t)
            idx = t == 1 ? 1 : tr[:noisy_sent=>t-1=>:idx]
            push!(inferred_idxs, idx)
            push!(inferred_words, idx <= length(sent) ? sent[idx] : "<none>")
            push!(inferred_actions, tr[:noisy_sent=>t=>:action])
            push!(obs_words, tr[:noisy_sent=>t=>:word])
        end
    end

    df = DataFrame(
        t = step,
        inferred_word = inferred_words,
        inferred_action = inferred_actions,
        inferred_idx = inferred_idxs,
        obs_word = obs_words,
    )
    CSV.write(join([output_dir, "trace_results.csv"], "/"), df)

    inferred_sents = []
    for tr in traces
        sent = tr[:intended_sent]
        sent = sent[end].context
        sent = join(sent, " ")
        push!(inferred_sents, sent)
    end

    df = DataFrame(inferred_sent = inferred_sents)
    CSV.write(join([output_dir, "inferred_sents.csv"], "/"), df)

    action_prob_dists = []
    for tr in traces
        (action_probs, sent) = tr[]
        push!(action_prob_dists, action_probs)
    end
    action_prob_arr = vcat(transpose(action_prob_dists)...)
    CSV.write(
        join([output_dir, "action_probs.csv"], "/"),
        DataFrame(action_prob_arr, ACTION_LIST),
    )

    params_dists = []
    for tr in traces
        param_posterior =
            [tr[:form_sub_param], tr[:sem_sub_param]]
        push!(params_dists, param_posterior)
    end
    param_dist_arr = vcat(transpose(params_dists)...)
    CSV.write(
        join([output_dir, "params_dists.csv"], "/"),
        DataFrame(
            param_dist_arr,
            ["form_sub_param", "sem_sub_param"],
        ),
    )

end

# for use in particle_filter_step
# given the previous trace, the current time step, and an observation, make random choices 
# that are consistent with the observation
@gen function choose_action(prev_trace, t::Int, obs::String)
    i = t > 1 ? prev_trace[:noisy_sent=>t-1=>:idx] : 1
    sent =
        length(prev_trace[:intended_sent]) == 0 ? String[] :
        copy(prev_trace[:intended_sent][end].context)

    # assuming the intended sentence RV does not yet contain anything based on the obs yet
    if !(Gen.has_value(Gen.get_choices(prev_trace), :intended_sent => i => :w)) &&
       (length(sent) == 0 || sent[end] ∉ EOS_TOKENS)

        if obs ∉ EOS_TOKENS && Gen.bernoulli(1 - prev_trace[:action_prior][1])
            if Gen.bernoulli(0.5)
                # Option A: Propose from the LM distribution given prior context, w/o peeking
                new_word = {:intended_sent => i => :w} ~ gpt_word_dist(sent)
                push!(sent, new_word)
            else
                # Option B: Propose based on similar words/neighbors to the observed word
                p_form = top_x_mask(get_form_sub_ps(obs, prev_trace[:form_sub_param]), 3)
                p_sem = top_x_mask(get_sem_sub_ps(obs, prev_trace[:sem_sub_param]), 3)
                new_word = {:intended_sent => i => :w} ~ word_mixture_dist([1.0, 1.0] ./ 2.0, p_form, p_sem)
                push!(sent, new_word)
            end
        else
            # Option C: Propose deterministically using the observed word
            new_word = {:intended_sent => i => :w} ~ word_dist(word_onehot(obs))
            push!(sent, new_word)
        end
    end

    # this logic ensures that the assigned action is compatible with the combination
    # of intended word and observed word
    if "disfl" in ACTION_LIST && obs in DISFL_LIST
        {:noisy_sent => t => :action} ~ action_dist(action_onehot("disfl"))
        {:noisy_sent => t => :idx} ~ dummy_dist(i)
    elseif "normal" in ACTION_LIST && (length(sent) >= i) && (obs == sent[i])
        {:noisy_sent => t => :action} ~ action_dist(action_onehot("normal"))
        i += 1
        {:noisy_sent => t => :idx} ~ dummy_dist(i)
    elseif "skip" in ACTION_LIST &&
           (length(sent) >= i + 1) &&
           (obs == sent[i+1]) &&
           Gen.bernoulli(0.9)
        {:noisy_sent => t => :action} ~ action_dist(action_onehot("skip"))
        i += 2
        {:noisy_sent => t => :idx} ~ dummy_dist(i)
    elseif "backtrack" in ACTION_LIST &&
           (length(sent) >= i - 1) &&
           (obs in sent[1:i-1]) &&
           Gen.bernoulli(0.5)
        {:noisy_sent => t => :action} ~ action_dist(action_onehot("backtrack"))
        i = findfirst(x -> x == obs, sent) + 1
        {:noisy_sent => t => :idx} ~ dummy_dist(i)
    elseif "morph_sub" in ACTION_LIST &&
           (length(sent) >= i) &&
           (get_morph_sub_ps(sent[i]) == word_onehot(obs))
        {:noisy_sent => t => :action} ~ action_dist(action_onehot("morph_sub"))
        i += 1
        {:noisy_sent => t => :idx} ~ dummy_dist(i)
    elseif length(sent) >= i
        if "sem_sub" in ACTION_LIST &&
           (get_sem_sub_ps(sent[i], prev_trace[:sem_sub_param])[get_vocab_idx(obs)] != 0) &&
           Gen.bernoulli(0.5)
            {:noisy_sent => t => :action} ~ action_dist(action_onehot("sem_sub"))
            i += 1
            {:noisy_sent => t => :idx} ~ dummy_dist(i)
        elseif "form_sub" in ACTION_LIST && (
            get_form_sub_ps(sent[i], prev_trace[:form_sub_param])[get_vocab_idx(obs)] != 0
        )
            {:noisy_sent => t => :action} ~ action_dist(action_onehot("form_sub"))
            i += 1
            {:noisy_sent => t => :idx} ~ dummy_dist(i)
        else
            {:noisy_sent => t => :action} ~ action_dist(action_onehot("insert"))
            {:noisy_sent => t => :idx} ~ dummy_dist(i)
        end
    elseif "insert" in ACTION_LIST
        {:noisy_sent => t => :action} ~ action_dist(action_onehot("insert"))
        {:noisy_sent => t => :idx} ~ dummy_dist(i)
    else
        print("choose_action cannot propose an action based on the provided ACTION_LIST")
    end
end

# a rejuvenation proposal function that proposes alternative random choices for a given trace
@gen function rejuv_proposal_add_delete(prev_trace, tt::Int)
    (T,) = Gen.get_args(prev_trace)

    old_idx = tt == 1 ? 1 : prev_trace[:noisy_sent=>tt-1=>:idx]

    # the observed words (same for all traces)
    literal_prefix = [prev_trace[:noisy_sent=>i=>:word] for i = 1:T]

    # randomness to decide whether we are generating an alternative or giving back the original (needed for involution to work)
    add = {:add} ~ Gen.bernoulli(0.5)

    new_sent = copy(prev_trace[:intended_sent][end].context)
    if add && (length(new_sent) >= old_idx - 1)
        word = gpt_word_dist(new_sent[1:old_idx-1])
        insert!(new_sent, old_idx, word)
    elseif length(new_sent) >= old_idx
        deleteat!(new_sent, old_idx)
    end

    # if the new sentence we are generating is longer than the old one then we may need to change the lookahead variable
    # lookahead determines how many words ahead of the current observation the LM samples (should be 0 or 1)
    lookahead = {:lookahead} ~ Gen.bernoulli(0.5)

    # if reverse=true, always just generate the original sentence
    # if reverse=false, always just generate the provided new sentence
    # a function call with reverse=false can be inverted if reverse=true
    # however a function call with reverse=true leaves a trace unchanged
    intended_sent =
        {:intended_sent} ~ generate_intended_sentence(
            T + lookahead,
            LMState(GPT_PROMPT, String[], new_sent, false),
        )
    sent = length(intended_sent) == 0 ? String[] : intended_sent[end].context

    state = ModelProposalState(
        prev_trace[:form_sub_param],
        prev_trace[:sem_sub_param],
        sent,
        literal_prefix,
        prev_trace[:noisy_sent=>tt=>:action],
        1,
        tt,
        add
    )
    noisy_sent ~ generate_noisy_sent_proposal_unfold(T, state)

    return prev_trace[:lookahead]
end

function involution_add_delete(tr, forward_choices, forward_retval, proposal_args)

    # forward  : run the (stochastic) proposal function on the current trace
    # backward : run the (stochastic) proposal function on the resultant trace from the forward pass

    # reverse: the proposal function does a coin-flip of whether it actually proposes the new sentence
    # or just proposes the same sentence from the old trace

    # old trace: the existing trace, before applying MH move
    (t,) = proposal_args

    new_trace_choices = Gen.choicemap()
    backward_choices = Gen.choicemap()

    backward_choices[:add] = !forward_choices[:add]

    new_trace_choices[:lookahead] = forward_choices[:lookahead]
    backward_choices[:lookahead] = forward_retval

    # we can use the Gen.set_submap!() function to change an entire subtree within the hierarchical random choices
    # then the Gen.update() step will compare trace probabilities but it's OK if the two traces don't share
    # all the same addresses
    Gen.set_submap!(new_trace_choices, :intended_sent, Gen.get_submap(forward_choices, :intended_sent))
    Gen.set_submap!(
        new_trace_choices,
        :noisy_sent,
        Gen.get_submap(forward_choices, :noisy_sent),
    )
    Gen.set_submap!(
        backward_choices,
        :intended_sent,
        Gen.get_submap(Gen.get_choices(tr), :intended_sent),
    )
    Gen.set_submap!(
        backward_choices,
        :noisy_sent,
        Gen.get_submap(Gen.get_choices(tr), :noisy_sent),
    )

    new_trace, weight, = Gen.update(tr, Gen.get_args(tr), (Gen.UnknownChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end

# rejuvenation proposal: form-based substitution
@gen function rejuv_proposal_form_sub(prev_trace, t::Int, sub_type::String)
    old_index = t == 1 ? 1 : prev_trace[:noisy_sent=>t-1=>:idx]
    ps = get_form_sub_ps(prev_trace[:intended_sent=>old_index=>:w], prev_trace[:form_sub_param])
    new_word = {:intended_sent => old_index => :w} ~ word_dist(ps)
    err_type = prev_trace[:noisy_sent=>t=>:action] == "normal" ? "form_sub" : "normal"
    new_action = {:noisy_sent => t => :action} ~ action_dist(action_onehot(err_type))
end

# rejuvenation proposal: semantic substitution
@gen function rejuv_proposal_sem_sub(prev_trace, t::Int, sub_type::String)
    old_index = t == 1 ? 1 : prev_trace[:noisy_sent=>t-1=>:idx]
    ps = get_sem_sub_ps(prev_trace[:intended_sent=>old_index=>:w], prev_trace[:sem_sub_param])
    new_word = {:intended_sent => old_index => :w} ~ word_dist(ps)
    err_type = prev_trace[:noisy_sent=>t=>:action] == "normal" ? "sem_sub" : "normal"
    new_action = {:noisy_sent => t => :action} ~ action_dist(action_onehot(err_type))
end

# rejuvenation proposal: morphological substitution
@gen function rejuv_proposal_morph_sub(prev_trace, t::Int, sub_type::String)
    old_index = t == 1 ? 1 : prev_trace[:noisy_sent=>t-1=>:idx]
    ps = get_morph_sub_ps(prev_trace[:intended_sent=>old_index=>:w])
    if sum(ps) == 1
        new_word = {:intended_sent => old_index => :w} ~ word_dist(ps)
    else
        new_word = {:intended_sent => old_index => :w} ~ nonword_dist()
    end
    err_type = prev_trace[:noisy_sent=>t=>:action] == "normal" ? "morph_sub" : "normal"
    new_action = {:noisy_sent => t => :action} ~ action_dist(action_onehot(err_type))
end

function involution_sub(tr, forward_choices, forward_retval, proposal_args)
    (T,) = Gen.get_args(tr)
    (t, sub_type) = proposal_args

    new_trace_choices = Gen.choicemap()
    backward_choices = Gen.choicemap()

    i = t == 1 ? 1 : tr[:noisy_sent=>t-1=>:idx]

    new_trace_choices[:intended_sent=>i=>:w] = forward_choices[:intended_sent=>i=>:w]
    new_trace_choices[:noisy_sent=>t=>:action] =
        (tr[:noisy_sent=>t=>:action] == "normal" ? sub_type : "normal")

    backward_choices[:intended_sent=>i=>:w] = tr[:intended_sent=>i=>:w]
    backward_choices[:noisy_sent=>t=>:action] = tr[:noisy_sent=>t=>:action]

    new_trace, weight, = Gen.update(tr, Gen.get_args(tr), (Gen.NoChange(),), new_trace_choices)
    (new_trace, backward_choices, weight)
end

function particle_filter_with_rejuv(
    num_particles::Int,
    utt::Vector{<:AbstractString},
    num_samples::Int,
    folder::String
)

    inferred_sentences_list = []
    log_weights_list = []
    inferred_actions_list = []

    # initialize empty choicemap
    init_obs = Gen.choicemap()

    # initialize the particle filter
    state = Gen.initialize_particle_filter(model_unfold, (0,), init_obs, num_particles)

    # Prepare a thread-local storage: one array per thread.
    local_results = [Vector{Dict{String,Any}}() for _ = 1:nthreads()]

    @printf("%-2s  %-15s %-15s %-15s %-15s\n", "", "Word", "Surprisal", "Unigram", "P(rejuv)")
    println("--------------------------------")

    for t = 1:length(utt)

        # adding one observation
        obs = Gen.choicemap((:noisy_sent => t => :word, utt[t]))

        # particle filter weight update
        (log_incr_weights,) = Gen.particle_filter_step!(
            state,
            (t,),
            (Gen.UnknownChange(),),
            obs,
            choose_action,
            (t, utt[t]),
        )

        # particle weights
        t_log_weights = copy(state.log_weights)
        push!(log_weights_list, t_log_weights)

        max_logw = maximum(t_log_weights)
        log_mean_weight = max_logw + log.(mean(exp.(t_log_weights .- max_logw)))

        # inferred actions
        push!(
            inferred_actions_list,
            [state.traces[i][:noisy_sent=>t=>:action] for i = 1:num_particles],
        )

        # inferred sentence
        counts = countmap([
            join(state.traces[i][:intended_sent][end].context, " ") for i = 1:num_particles
        ])
        push!(inferred_sentences_list, findmax(counts)[2])

        # trying a change to the resample location
        # after resampling, all log weights will be set to 0 (uniform across particles)
        resampled = Gen.maybe_resample!(state, ess_threshold = ESS_THRESH)

        surprisal = -log_mean_weight
        unigram_surp = -log(unigram_probs[get_vocab_idx(utt[t])])
        cond_rejuv_p = custom_sigmoid(surprisal - unigram_surp, args["logprob_thresh"], args["logprob_spread"])
        @printf("%-2s: %-15s %-15.2f %-15.2f %-15.2f\n", t, utt[t], surprisal, unigram_surp, cond_rejuv_p)

        # Conditional Reanalysis
        @threads for i = 1:num_particles
            args["conditional_rejuv"] || break

            # higher surprisal -> higher probability of applying rejuvenation
            Gen.bernoulli(cond_rejuv_p) || continue

            timesteps = max(1, t - args["lookback"]):t
            steps =
                (REJUVENATE_ORDER == "BACKWARD") ? reverse(timesteps) :
                ((REJUVENATE_ORDER == "FORWARD") ? timesteps : shuffle(timesteps))

            for tt in steps
                index_back = tt == 1 ? 1 : state.traces[i][:noisy_sent=>tt-1=>:idx]

                # SUBSTITUTIONS
                if Gen.has_value(
                    Gen.get_choices(state.traces[i]),
                    :intended_sent => index_back => :w,
                ) && state.traces[i][:intended_sent=>index_back=>:w] != "<nonword>"

                    # Form Sub
                    state.traces[i], accepted = Gen.mh(
                        state.traces[i],
                        rejuv_proposal_form_sub,
                        (tt, "form_sub"),
                        involution_sub,
                    )
                    log_rejuv_result!(local_results, t, i, "sub_error", accepted, tt)

                    # Semantic Sub
                    state.traces[i], accepted = Gen.mh(
                        state.traces[i],
                        rejuv_proposal_sem_sub,
                        (tt, "sem_sub"),
                        involution_sub,
                    )
                    log_rejuv_result!(local_results, t, i, "sub_error", accepted, tt)

                    # Morphological Sub
                    state.traces[i], accepted = Gen.mh(
                        state.traces[i],
                        rejuv_proposal_morph_sub,
                        (tt, "morph_sub"),
                        involution_sub,
                    )
                    log_rejuv_result!(local_results, t, i, "sub_error", accepted, tt)

                    # Insertions & Skips
                    state.traces[i], accepted = Gen.mh(
                        state.traces[i],
                        rejuv_proposal_add_delete,
                        (tt,),
                        involution_add_delete,
                    )
                    log_rejuv_result!(local_results, t, i, "insert_error", accepted, tt)
                end

                # ACTION PRIOR and ACTION ALPHAS
                state.traces[i], accepted =
                    Gen.mh(state.traces[i], Gen.select(:action_prior))
                log_rejuv_result!(local_results, t, i, "action_prior", accepted, tt)

                # SUBSTITUTION PARAMETERS
                state.traces[i], accepted =
                    Gen.mh(state.traces[i], Gen.select(:form_sub_param, :sem_sub_param))
                log_rejuv_result!(local_results, t, i, "substitution_temp", accepted, tt)
            end
        end

    end

    # SECOND-PASS REJUVENATION
    for j = 1:args["second_pass_rejuv_iters"]
        for i = 1:num_particles

            args["second_pass_rejuv"] || break

            Gen.bernoulli(args["second_pass_rejuv_p"]) || continue

            timesteps = 1:length(utt)
            steps =
                (REJUVENATE_ORDER == "BACKWARD") ? reverse(timesteps) :
                ((REJUVENATE_ORDER == "FORWARD") ? timesteps : shuffle(timesteps))

            for tt in steps

                index_back = tt == 1 ? 1 : state.traces[i][:noisy_sent=>tt-1=>:idx]

                # SUBSTITUTIONS
                if Gen.has_value(Gen.get_choices(state.traces[i]), :intended_sent => index_back => :w) &&
                   state.traces[i][:intended_sent=>index_back=>:w] != "<nonword>"

                    # Form Sub
                    state.traces[i], accepted = Gen.mh(
                        state.traces[i],
                        rejuv_proposal_form_sub,
                        (tt, "form_sub"),
                        involution_sub,
                    )
                    log_rejuv_result!(local_results, length(utt), i, "sub_error", accepted, tt)

                    # Semantic Sub
                    state.traces[i], accepted = Gen.mh(
                        state.traces[i],
                        rejuv_proposal_sem_sub,
                        (tt, "sem_sub"),
                        involution_sub,
                    )
                    log_rejuv_result!(local_results, length(utt), i, "sub_error", accepted, tt)

                    # Morphological Sub
                    state.traces[i], accepted = Gen.mh(
                        state.traces[i],
                        rejuv_proposal_morph_sub,
                        (tt, "morph_sub"),
                        involution_sub,
                    )
                    log_rejuv_result!(local_results, length(utt), i, "sub_error", accepted, tt)

                    # Insertions & Skips
                    state.traces[i], accepted = Gen.mh(
                        state.traces[i],
                        rejuv_proposal_add_delete,
                        (tt,),
                        involution_add_delete,
                    )
                    log_rejuv_result!(local_results, length(utt), i, "insert_error", accepted, tt)
                end

                # ACTION PRIOR and ACTION ALPHAS
                state.traces[i], accepted = Gen.mh(state.traces[i], Gen.select(:action_prior))
                log_rejuv_result!(local_results, length(utt), i, "action_prior", accepted, tt)

                # SUBSTITUTION PARAMETERS
                state.traces[i], accepted =
                    Gen.mh(state.traces[i], Gen.select(:form_sub_param, :sem_sub_param))
                log_rejuv_result!(local_results, length(utt), i, "substitution_temp", accepted, tt)
            end
        end
    end

    # the weight of each particle at each time step
    log_weights_arr = hcat(log_weights_list...)
    CSV.write("$folder/log_weights.csv", DataFrame(log_weights_arr, :auto))

    # output the number of accepted rejuvenation moves at each step 
    rejuvenation_data = vcat(local_results...)
    rejuvenation_df = DataFrame(rejuvenation_data)
    CSV.write("$folder/acceptances_by_t.csv", rejuvenation_df)

    # incremental inferred actions
    inferred_actions_df = DataFrame(hcat(inferred_actions_list...), :auto)
    CSV.write("$folder/inferred_actions.csv", inferred_actions_df)

    # incremental inferred sentences
    inferred_sentences_df = DataFrame(top_inferred_sentence = inferred_sentences_list)
    CSV.write("$folder/top_inferred_sentences_by_t.csv", inferred_sentences_df)

    # return a sample of unweighted traces from the weighted collection
    return Gen.sample_unweighted_traces(state, num_samples)
end;

function test_sent(test_sents_file, sent_id, folder)

    test_sentence = split(rstrip(read(test_sents_file, String)), "\n")[sent_id]
    println("Inference for utterance: $test_sentence")
    test_sentence_split = String.(split(test_sentence))

    # create directiories if they don't exist yet
    filepath = "$folder/$sent_id"
    isdir(filepath) || mkpath(filepath)

    @time pf_traces = particle_filter_with_rejuv(
        args["num_particles"],
        test_sentence_split,
        args["num_samples"],
        "$folder/$sent_id",
    )
    analyze_traces(pf_traces, length(test_sentence_split), "$folder/$sent_id")

    # LM only surprisal
    # make a choicemap corresponding to words in the test utterance
    choices = Gen.choicemap()
    for (j, u) in enumerate(test_sentence_split)
        choices[j=>:w] = u
    end

    # condition the LM model on generating the test utterance
    (tr, weight) = Gen.generate(
        generate_intended_sentence,
        (length(test_sentence_split), LMState(GPT_PROMPT, String[], String[], false)),
        choices,
    )

    # get the scores for words in the utterance
    scores = [Gen.project(tr, Gen.select(k => :w)) for k = 1:length(test_sentence_split)]
    df = DataFrame(surprisal = -1 .* scores, index = 1:length(test_sentence_split))
    CSV.write("$folder/$sent_id/lm_only_surps.csv", df)
end

# run particle filter inference for a single test sentence
function debug_one()

    input_sentence = args["debug_input"]

    println("Inference for utterance: $input_sentence")
    words = String.(split(input_sentence))

    filepath = "$BASE_DIRECTORY/log/debug"
    isdir(filepath) || mkpath(filepath)

    @time pf_traces = particle_filter_with_rejuv(
        args["num_particles"],
        words,
        args["num_samples"],
        filepath,
    )
    analyze_traces(pf_traces, length(words), filepath)

    # LM only surprisal: make a choicemap corresponding to words in the test utterance
    choices = Gen.choicemap()
    for (j, u) in enumerate(words)
        choices[j=>:w] = u
    end

    # condition the LM model on generating the test utterance
    (tr, weight) = Gen.generate(
        generate_intended_sentence,
        (length(words), LMState(GPT_PROMPT, String[], String[], false)),
        choices,
    )

    # get the scores for words in the utterance
    scores = [Gen.project(tr, Gen.select(k => :w)) for k = 1:length(words)]
    df = DataFrame(surprisal = -1 .* scores, index = 1:length(words))
    CSV.write("$BASE_DIRECTORY/log/debug/lm_only_surps.csv", df)
end


# MAIN
if args["mode"] == "test_one"
    test_sent(args["test_sents_file"], args["sent_id"], args["output_dir"])
elseif args["mode"] == "debug_one"
    debug_one()
else
    println("Invalid mode option")
end

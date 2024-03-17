def generate_base_name(params: dict):
    proj_params = params.get('projection')
    optim_params = params.get('optimizer')
    return (
        f"{proj_params.get('proj_type')}-"
        f"{params.get('environment').get('env_id')}-"
        f"{params.get('policy').get('policy_type')}-"
        f"{'CONTEXT-' if params.get('policy').get('contextual_std') else ''}"
        f"m{proj_params.get('mean_bound')}-"
        f"c{proj_params.get('cov_bound')}-"
        f"{'e' + str(proj_params.get('target_entropy')) + '-' if proj_params.get('entropy_schedule') else ''}"
        f"{'_' + str(proj_params.get('entropy_schedule')) + '-' if proj_params.get('entropy_schedule') else ''}"
        f"{'first' + str(proj_params.get('entropy_first')) + '-' if proj_params.get('entropy_schedule') else ''}"
        f"{'eq' + str(proj_params.get('entropy_eq')) + '-' if proj_params.get('entropy_schedule') else ''}"
        f"{'temp' + str(proj_params.get('temperature')) + '-' if proj_params.get('entropy_schedule') else ''}"
        f"{'lr_reg' + str(proj_params.get('lr_reg')) + '-' if proj_params.get('do_regression') else ''}"
        f"{'delta' + str(proj_params.get('trust_region_coeff')) + '-' if proj_params.get('trust_region_coeff') else ''}"
        f"{'schedule' + str(optim_params.get('lr_schedule')) + '-' if optim_params.get('lr_schedule') else ''}"
        f"lr_policy{optim_params.get('lr_policy')}-"
        f"lr_critic{optim_params.get('lr_critic')}-"
    )


def generate_pg_exp_id(params: dict):
    return (
        f"{generate_base_name(params)}"
        f"{'clip' + str(params.get('algorithm').get('importance_ratio_clip')) + '-' if params.get('algorithm').get('importance_ratio_clip') else ''}"
        f"{'max_ent' + str(params.get('algorithm').get('max_entropy_coeff')) + '-' if params.get('algorithm').get('max_entropy_coeff') else ''}"
        # f"obs{params.get('norm_observations')}-"
        f"{'discount' + str(params.get('algorithm').get('discount_factor')) + '-' if params.get('algorithm').get('discount_factor') != 0.99 else ''}"
        f"{str(params.get('exp_name')) + '-' if params.get('exp_name') else ''}"
        f"mod{str(params.get('environment').get('replanning_interval')) + '-' if params.get('environment').get('replanning_interval', -1) > 0 else ''}"
        f"steps{params.get('training').get('train_steps')}-"
        f"epochs{params.get('training').get('epochs')}-{params.get('training').get('epochs_critic')}-"
        f"n_minibatches{params.get('training').get('n_minibatches')}-"
        f"samples{params.get('training').get('n_training_samples')}-"
        f"n_envs{params.get('environment').get('n_envs')}-"
        f"p{params.get('policy').get('hidden_sizes')}-"
        f"c{params.get('critic').get('hidden_sizes')}-"
        f"seed{params.get('seed')}"
    )


def generate_sac_exp_id(params: dict):
    return (
        f"{generate_base_name(params)}"
        f"lr_alpha{params.get('optimizer').get('lr_alpha')}-"
        f"alpha{params.get('algorithm').get('alpha')}-"
        # f"cov{params.get('init_std')}-"
        # f"min_std{params.get('minimal_std')}-"
        f"{str(params.get('exp_name')) + '-' if params.get('exp_name') else ''}"
        f"steps{params.get('training').get('train_steps')}-"
        f"updates{params.get('training').get('updates_per_epoch')}-"
        f"batches{params.get('training').get('batch_size')}-"
        f"seed{params.get('seed')}"
    )


def generate_td3_exp_id(params: dict):
    return (
        f"{generate_base_name(params)}"
        f"cov{params.get('algorithm').get('exploration_noise')}-"
        f"{str(params.get('exp_name')) + '-' if params.get('exp_name') else ''}"
        f"steps{params.get('training').get('train_steps')}-"
        f"updates{params.get('training').get('updates_per_epoch')}-"
        f"batches{params.get('training').get('batch_size')}-"
        f"seed{params.get('seed')}"
    )


def generate_mpo_exp_id(params: dict):
    return (
        f"{generate_base_name(params)}"
        f"lr_dual{params.get('optimizer').get('lr_dual')}-"
        f"eps{params.get('algorithm').get('dual_constraint')}-"
        f"eps_mu{params.get('algorithm').get('mean_constraint')}-"
        f"eps_sig{params.get('algorithm').get('var_constraint')}-"
        f"eta{params.get('algorithm').get('log_eta')}-"
        f"alpha_mu{params.get('algorithm').get('log_alpha_mu')}-"
        f"alpha_std{params.get('algorithm').get('log_alpha_std')}-"
        f"{str(params.get('exp_name')) + '-' if params.get('exp_name') else ''}"
        f"steps{params.get('training').get('train_steps')}-"
        f"updates{params.get('training').get('updates_per_epoch')}-"
        f"epochs{params.get('training').get('batch_size')}-"
        f"seed{params.get('seed')}"
    )


def generate_vlearn_exp_id(params: dict):
    return (
        f"{generate_base_name(params)}"
        # f"lr_alpha{params.get('optimizer').get('lr_alpha')}-"
        f"alpha{params.get('algorithm').get('alpha')}-"
        f"ent{params.get('algorithm').get('entropy_coeff')}-"
        # f"cov{params.get('init_std')}-"
        f"min_std{params.get('policy').get('minimal_std')}-"
        f"{str(params.get('exp_name')) + '-' if params.get('exp_name') else ''}"
        f"steps{params.get('training').get('train_steps')}-"
        f"updates{params.get('training').get('updates_per_epoch')}-"
        f"epochs{params.get('training').get('batch_size')}-"
        # f"trl_{params.get('algorithm').get('trl_policy_update')}-"
        # f"log_{params.get('algorithm').get('log_policy_update')}"
        f"freq{params.get('training').get('sample_frequency')}-"
        f"log_clip{params.get('algorithm').get('log_ratio_clip')}-"
        # f"p_intv{params.get('algorithm').get('policy_target_update_interval')}-"
        f"norm{params.get('algorithm').get('advantage_norm')}-"
        # f"grad_clip{params.get('optimizer').get('clip_grad_norm')}-"
        # f"v_loss{params.get('value_loss').get('value_loss_type')}-"
        # f"buffer{params.get('replay_buffer').get('buffer_type')}-"
        f"{'avg' + str(params.get('replay_buffer').get('polyak_weight')) if params.get('algorithm').get('log_policy_update') == 'avg' else ''}-"
        f"{'poly_trl' + str(params.get('algorithm').get('polyak_weight_policy_trl')) if params.get('algorithm').get('trl_policy_update') == 'polyak' else ''}-"
        f"{'poly_log' + str(params.get('algorithm').get('polyak_weight_policy_log')) if params.get('algorithm').get('log_policy_update') == 'polyak' else ''}-"
        f"p{params.get('policy').get('hidden_sizes')}-"
        f"c{params.get('critic').get('hidden_sizes')}-"
        f"buffer{params.get('replay_buffer').get('max_replay_buffer_size')}-"
        f"seed{params.get('seed')}"
    )


def generate_vlearnq_exp_id(params: dict):
    return (
        f"{generate_base_name(params)}"
        f"lr_alpha{params.get('optimizer').get('lr_alpha')}-"
        f"alpha{params.get('algorithm').get('alpha')}-"
        # f"cov{params.get('init_std')}-"
        # f"min_std{params.get('minimal_std')}-"
        f"{str(params.get('exp_name')) + '-' if params.get('exp_name') else ''}"
        f"steps{params.get('training').get('train_steps')}-"
        f"updates{params.get('training').get('updates_per_epoch')}-"
        f"batch{params.get('training').get('batch_size')}-"
        f"{'poly_trl' + str(params.get('algorithm').get('polyak_weight_policy_trl')) if params.get('algorithm').get('trl_policy_update') == 'polyak' else ''}-"
        f"{'poly_log' + str(params.get('algorithm').get('polyak_weight_policy_log')) if params.get('algorithm').get('log_policy_update') == 'polyak' else ''}-"
        f"buffer{params.get('replay_buffer').get('max_replay_buffer_size')}-"
        f"seed{params.get('seed')}"
    )


def generate_vtrace_exp_id(params: dict):
    return (
        f"{generate_base_name(params)}"
        f"{str(params.get('exp_name')) + '-' if params.get('exp_name') else ''}"
        f"steps{params.get('training').get('train_steps')}-"
        f"updates{params.get('training').get('updates_per_epoch')}-"
        f"batch{params.get('training').get('batch_size')}-"
        f"samples{params.get('training').get('n_training_samples')}-"
        f"ent_coef{params.get('algorithm').get('entropy_coeff')}-"
        f"buffer{params.get('replay_buffer').get('max_replay_buffer_size')}-"
        # f"p{params.get('policy').get('hidden_sizes')}-"
        # f"c{params.get('critic').get('hidden_sizes')}-"
        f"seed{params.get('seed')}"
    )


def generate_awr_exp_id(params: dict):
    return (
        f"{generate_base_name(params)}"
        f"{str(params.get('exp_name')) + '-' if params.get('exp_name') else ''}"
        f"steps{params.get('training').get('train_steps')}-"
        f"epoch{params.get('training').get('policy_epoch_steps')}-"
        f"epochs{params.get('training').get('critic_epoch_steps')}-"
        f"max{params.get('algorithm').get('max_weight')}-"
        f"temp{params.get('algorithm').get('beta')}-"
        f"log_clip{params.get('algorithm').get('log_clip')}-"
        f"n_step{params.get('replay_buffer').get('n_step')}_{params.get('replay_buffer').get('period_length')}"
        f"std{params.get('policy').get('init_std')}-"
        # f"p{params.get('policy').get('hidden_sizes')}-"
        # f"c{params.get('critic').get('hidden_sizes')}-"
        f"seed{params.get('seed')}"
    )


def generate_deep_pro_mp_exp_id(params: dict):
    return (
        f"{generate_base_name(params)}"
        f"{'clip' + str(params.get('algorithm').get('importance_ratio_clip')) + '-' if params.get('algorithm').get('importance_ratio_clip') else ''}"
        f"{'max_ent' + str(params.get('algorithm').get('max_entropy_coeff')) + '-' if params.get('algorithm').get('max_entropy_coeff') else ''}"
        # f"obs{params.get('norm_observations')}-"
        f"{str(params.get('exp_name')) + '-' if params.get('exp_name') else ''}"
        f"steps{params.get('training').get('train_steps')}-"
        f"epochs{params.get('training').get('epochs')}-"
        f"n_minibatches{params.get('training').get('n_minibatches')}-"
        f"samples{params.get('training').get('n_training_samples')}-"
        f"n_basis{params.get('pro_mp').get('n_basis')}"
        f"seed{params.get('seed')}"
    )

import git
import os
from cox.store import schema_from_dict

from trust_region_projections.algorithms.awr.awr import AWR
from trust_region_projections.algorithms.deep_pro_mp.deep_pro_mp import DeepProMP
from trust_region_projections.algorithms.mpo.mpo import MPO
from trust_region_projections.algorithms.vlearn.vlearn import VLearning
from trust_region_projections.algorithms.pg.pg import PolicyGradient
from trust_region_projections.algorithms.sac.sac import SAC
from trust_region_projections.algorithms.td3.td3 import TD3
from trust_region_projections.algorithms.vlearn_q.vlearn_q import VlearnQ
from trust_region_projections.utils.custom_store import CustomStore
from trust_region_projections.algorithms.vtrace.vtrace import Vtrace

BASE_ALGOS = ['ppo', 'papi', 'sac', 'td3', 'mpo', 'vlearn', 'vtrace', 'awr']


def setup_general_agent(params, save_git=False):
    # Do some sanity checks first to avoid wasting compute for stupid HP choices

    for k, v in zip(params.keys(), params.values()):
        assert v is not None, f"Value for {k} is None"

    pparams = params['projection']
    # ensure when not using entropy constraint, the cov is not projected to -inf by accident
    if not pparams.get('entropy_schedule'):
        params['entropy_eq'] = False

    # if pparams['proj_type'] not in BASE_ALGOS and pparams['trust_region_coeff'] == 0:
    #     # must be a projection layer with no trust region penalty at this point (typically that does not work)
    #     raise ValueError(f"Using projection {pparams['proj_type']} with penalty weight "
    #                      f"alpha={pparams['trust_region_coeff']} does not enforce the trust region "
    #                      f"and should not be used.")

    ####################################################################################################################
    store = None

    if params['logging']['log_interval'] <= params['training']['train_steps']:
        # Setup logging
        metadata_schema = schema_from_dict(params)
        base_directory = params['out_dir']
        exp_name = params.get('exp_name')

        store = CustomStore(storage_folder=base_directory, exp_id=exp_name, new=True)

        # Store the experiment path
        metadata_schema.update({'store_path': str})
        metadata_table = store.add_table('metadata', metadata_schema)
        metadata_table.update_row(params)
        metadata_table.update_row({
            'store_path': store.path,
        })

        if save_git:
            # the git commit for this experiment
            metadata_schema.update({'git_commit': str})
            repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)), search_parent_directories=True)
            metadata_table.update_row({'git_commit': repo.head.object.hexsha})

        metadata_table.flush_row()

        # use 0 for saving last model only,
        # use -1 for no saving at all
        if params['logging']['save_interval'] == 0:
            params['logging']['save_interval'] = params['training']['train_steps']

    return store


def get_pg_agent(params, save_git=False):
    store = setup_general_agent(params, save_git)

    if store:
        # Table for checkpointing models and envs
        if params['logging']['save_interval'] > 0:

            checkpoints_dict = {
                'policy': store.PYTORCH_STATE,
                'envs': store.PICKLE,
                'envs_test': store.PICKLE,
                # 'sampler': store.PICKLE,
                'optimizer_policy': store.PYTORCH_STATE,
                'iteration': int
            }

            if not params['policy']['share_weights'] and params['algorithm']["critic_coeff"] == 0:
                checkpoints_dict.update({'critic': store.PYTORCH_STATE,
                                         'optimizer_critic': store.PYTORCH_STATE
                                         })

            store.add_table('checkpoints', checkpoints_dict)
    else:
        store = None

    return PolicyGradient.agent_from_params(params, store=store)


def get_sac_agent(params, save_git=False):
    store = setup_general_agent(params, save_git)

    if store:
        # Table for checkpointing models and envs
        if params['logging']['save_interval'] > 0:
            checkpoints_dict = {
                'critic': store.PYTORCH_STATE,
                'policy': store.PYTORCH_STATE,
                'envs': store.PICKLE,
                'envs_test': store.PICKLE,
                # 'sampler': store.PICKLE,
                'optimizer_policy': store.PYTORCH_STATE,
                'optimizer_critic': store.PYTORCH_STATE,
                'iteration': int
            }
            if params['algorithm']['alpha'] == "auto":
                checkpoints_dict.update({
                    'log_alpha': store.PYTORCH_STATE,
                    'optimizer_alpha': store.PYTORCH_STATE,
                })
            store.add_table('checkpoints', checkpoints_dict)
    else:
        store = None

    return SAC.agent_from_params(params, store=store)


def get_td3_agent(params, save_git=True):
    store = setup_general_agent(params, save_git)

    if store:
        # Table for checkpointing models and envs
        if params['logging']['save_interval'] > 0:
            checkpoints_dict = {
                'critic': store.PYTORCH_STATE,
                'policy': store.PYTORCH_STATE,
                'policy_target': store.PYTORCH_STATE,
                'envs': store.PICKLE,
                'envs_test': store.PICKLE,
                # 'sampler': store.PICKLE,
                'optimizer_policy': store.PYTORCH_STATE,
                'optimizer_critic': store.PYTORCH_STATE,
                'iteration': int
            }

            store.add_table('checkpoints', checkpoints_dict)
    else:
        store = None

    return TD3.agent_from_params(params, store=store)


def get_mpo_agent(params, save_git=True):
    store = setup_general_agent(params, save_git)

    if store:
        # Table for checkpointing models and envs
        if params['logging']['save_interval'] > 0:
            checkpoints_dict = {
                'critic': store.PYTORCH_STATE,
                'policy': store.PYTORCH_STATE,
                'policy_target': store.PYTORCH_STATE,
                'log_eta': store.PYTORCH_STATE,
                'log_alpha_mu': store.PYTORCH_STATE,
                'log_alpha_sigma': store.PYTORCH_STATE,
                'envs': store.PICKLE,
                'envs_test': store.PICKLE,
                # 'sampler': store.PICKLE,
                'optimizer_policy': store.PYTORCH_STATE,
                'optimizer_critic': store.PYTORCH_STATE,
                'optimizer_dual': store.PYTORCH_STATE,
                'iteration': int
            }

            store.add_table('checkpoints', checkpoints_dict)
    else:
        store = None

    return MPO.agent_from_params(params, store=store)


def get_vlearn_agent(params, save_git=False):
    store = setup_general_agent(params, save_git)

    if store:
        # Table for checkpointing models and envs
        if params['logging']['save_interval'] > 0:
            checkpoints_dict = {
                'critic': store.PYTORCH_STATE,
                'policy': store.PYTORCH_STATE,
                'envs': store.PICKLE,
                'envs_test': store.PICKLE,
                # 'sampler': store.PICKLE,
                'optimizer_policy': store.PYTORCH_STATE,
                'optimizer_critic': store.PYTORCH_STATE,
                'iteration': int
            }
            if params['algorithm']['alpha'] == "auto":
                checkpoints_dict.update({
                    'log_alpha': store.PYTORCH_STATE,
                    'optimizer_alpha': store.PYTORCH_STATE,
                })
            store.add_table('checkpoints', checkpoints_dict)
    else:
        store = None

    return VLearning.agent_from_params(params, store=store)


def get_vlearnq_agent(params, save_git=False):
    store = setup_general_agent(params, save_git)

    if store:
        # Table for checkpointing models and envs
        if params['logging']['save_interval'] > 0:
            checkpoints_dict = {
                'critic': store.PYTORCH_STATE,
                'policy': store.PYTORCH_STATE,
                'envs': store.PICKLE,
                'envs_test': store.PICKLE,
                # 'sampler': store.PICKLE,
                'optimizer_policy': store.PYTORCH_STATE,
                'optimizer_critic': store.PYTORCH_STATE,
                'iteration': int
            }
            if params['algorithm']['alpha'] == "auto":
                checkpoints_dict.update({
                    'log_alpha': store.PYTORCH_STATE,
                    'optimizer_alpha': store.PYTORCH_STATE,
                })
            store.add_table('checkpoints', checkpoints_dict)
    else:
        store = None

    return VlearnQ.agent_from_params(params, store=store)


def get_vtrace_agent(params, save_git=False):
    store = setup_general_agent(params, save_git)

    if store:
        # Table for checkpointing models and envs
        if params['logging']['save_interval'] > 0:
            checkpoints_dict = {
                'critic': store.PYTORCH_STATE,
                'policy': store.PYTORCH_STATE,
                'envs': store.PICKLE,
                'envs_test': store.PICKLE,
                'optimizer_policy': store.PYTORCH_STATE,
                'optimizer_critic': store.PYTORCH_STATE,
                'iteration': int
            }

            store.add_table('checkpoints', checkpoints_dict)
    else:
        store = None

    return Vtrace.agent_from_params(params, store=store)


def get_awr_agent(params, save_git=False):
    store = setup_general_agent(params, save_git)

    if store:
        # Table for checkpointing models and envs
        if params['logging']['save_interval'] > 0:
            checkpoints_dict = {
                'critic': store.PYTORCH_STATE,
                'policy': store.PYTORCH_STATE,
                'envs': store.PICKLE,
                'envs_test': store.PICKLE,
                'optimizer_policy': store.PYTORCH_STATE,
                'optimizer_critic': store.PYTORCH_STATE,
                'iteration': int
            }

            store.add_table('checkpoints', checkpoints_dict)
    else:
        store = None

    return AWR.agent_from_params(params, store=store)


def get_deep_pro_mp_agent(params, save_git=True):
    store = setup_general_agent(params, save_git)

    if store:
        # Table for checkpointing models and envs
        if params['logging']['save_interval'] > 0:

            checkpoints_dict = {
                'policy': store.PYTORCH_STATE,
                'envs': store.PICKLE,
                'envs_test': store.PICKLE,
                'optimizer_policy': store.PYTORCH_STATE,
                'optimizer_critic': store.PYTORCH_STATE,
                'iteration': int
            }

            if not params['policy']['share_weights'] and params['algorithm']["critic_coeff"] == 0:
                checkpoints_dict.update({'critic': store.PYTORCH_STATE,
                                         'optimizer_critic': store.PYTORCH_STATE
                                         })

            store.add_table('checkpoints', checkpoints_dict)
    else:
        store = None

    return DeepProMP.agent_from_params(params, store=store)

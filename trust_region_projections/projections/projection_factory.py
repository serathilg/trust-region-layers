from trust_region_projections.projections.base_projection_layer import BaseProjectionLayer
from trust_region_projections.projections.frob_projection_layer import FrobeniusProjectionLayer
from trust_region_projections.projections.kl_projection_layer import KLProjectionLayer
from trust_region_projections.projections.papi_projection import PAPIProjection
from trust_region_projections.projections.w2_projection_layer import WassersteinProjectionLayer
from trust_region_projections.projections.w2_projection_layer_non_com import \
    WassersteinProjectionLayerNonCommuting


def get_projection_layer(proj_type: str = "", **kwargs) -> BaseProjectionLayer:
    """
    Factory to generate the projection layers for all projections.
    Args:
        proj_type:
        **kwargs:

    Returns:

    """
    if not proj_type or proj_type.isspace() or proj_type.lower() in ["ppo", "sac", "td3", "mpo", "vlearn", "vtrace",
                                                                     "awr", "entropy"]:
        return BaseProjectionLayer(proj_type, **kwargs)

    elif proj_type.lower() == "w2":
        return WassersteinProjectionLayer(proj_type, **kwargs)

    elif proj_type.lower() == "w2_non_com":
        return WassersteinProjectionLayerNonCommuting(proj_type, **kwargs)

    elif proj_type.lower() == "frob":
        return FrobeniusProjectionLayer(proj_type, **kwargs)

    elif proj_type.lower() == "kl":
        return KLProjectionLayer(proj_type, **kwargs)

    elif proj_type.lower() == "papi":
        # papi has a different approach compared to our projections.
        # It has to be applied after the training with PPO.
        return PAPIProjection(proj_type, **kwargs)

    else:
        raise ValueError(
            f"Invalid projection type {proj_type}."
            f" Choose one of None/' ', 'ppo', 'sac', 'td3', 'mpo', 'vtrace',"
            f" 'papi', 'w2', 'w2_non_com', 'frob', 'kl', or 'entropy'.")

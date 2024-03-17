try:
    from GPUtil import GPUtil
except ModuleNotFoundError:
    pass


def get_free_gpus():
    try:
        return GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.8, maxMemory=0.8, includeNan=False,
                               excludeID=[], excludeUUID=[])
    except NameError:
        return []

from enum import Enum


class Policy(Enum):
    gEDF = 0
    apedf_ff = 1
    apedf_wf = 2
    a2pedf_ff = 3
    a2pedf_wf = 4
    a2pedf_wf_0 = 5
    a2pedf_wf_1 = 6

    @staticmethod
    def from_str(policy_str):
        if policy_str == 'gEDF':
            return Policy.gEDF
        elif policy_str == 'apEDF-FF':
            return Policy.apedf_ff
        elif policy_str == 'apEDF-WF':
            return Policy.apedf_wf
        elif policy_str == 'a2pEDF-FF' or policy_str == 'a$^2$pEDF-FF' or policy_str == 'apedf-ff-f2gedf':
            return Policy.a2pedf_ff
        elif policy_str == 'a2pEDF-WF' or policy_str == "a2pedf-wf" or policy_str == 'a$^2$pEDF-WF' or  policy_str == 'a2pedf-ff-f2gedf':
            return Policy.a2pedf_wf
        elif policy_str == 'a2pedf-wf_0' or policy_str == 'a2pEDF-WF_0' or policy_str == 'a$^2$pEDF-WF_0':
            return Policy.a2pedf_wf_0
        elif policy_str == 'a2pedf-wf_1' or policy_str == 'a2pEDF-WF_1' or policy_str == 'a$^2$pEDF-WF_1':
            return Policy.a2pedf_wf_1
        else:
            raise ValueError("policy " + policy_str + " not known.")
from gps.algorithm.policy_opt.policy_opt import PolicyOpt
from gps.algorithm.policy.policy import Policy
import pdb
import numpy as np
import pickle
from std_msgs.msg import String 
from gps_dnn_policy_training_and_testing_pkg.CONSTANT import training_request_topic, training_response_topic
import rospy
import tempfile

class OutsourcePolicyOptToROS(PolicyOpt):
    def __init__(self, hyperparams, dO, dU):
        config = hyperparams
        PolicyOpt.__init__(self, config, dO, dU)
        self.pub = rospy.Publisher(training_request_topic, String)
        pass

    def update(self, obs, tgt_mu, tgt_prc, tgt_wt):
        N, T = obs.shape[:2]
        dU, dO = self._dU, self._dO

        obs = np.reshape(obs, (N*T, dO))
        tgt_mu = np.reshape(tgt_mu, (N*T, dU))
        tgt_prc = np.reshape(tgt_prc, (N*T, dU, dU))
        tgt_wt = np.reshape(tgt_wt, (N*T, 1, 1))

        while not rospy.is_shutdown() and self.pub.get_num_connections() == 0:
            rospy.sleep(1)

        req = {
            'obs': obs,
            'tgt_mu': tgt_mu,
            'tgt_prc': tgt_prc,
            'tgt_wt': tgt_wt,
        }

        f = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        pickle.dump(req, f)
        f.close()

        self.pub.publish(String(data=f.name))

        rospy.loginfo("OutsourcePolicyOptToROS reqeust sent")
        rospy.loginfo("OutsourcePolicyOptToROS waiting for response")
        resp = rospy.wait_for_message(training_response_topic, String)
        rospy.loginfo("OutsourcePolicyOptToROS got response")

        with open(resp.data, 'rb') as f:
            policy = pickle.load(f)

        self.policy = policy
        return policy

    def prob(self, obs):
        dU = self._dU
        N, T = obs.shape[:2]
        output = np.zeros((N, T, dU))
        self.var = np.ones(dU)
        pol_sigma = np.tile(np.diag(self.var), [N, T, 1, 1])
        pol_prec = np.tile(np.diag(1.0 / self.var), [N, T, 1, 1])
        pol_det_sigma = np.tile(np.prod(self.var), [N, T])

        return output, pol_sigma, pol_prec, pol_det_sigma

    def set_ent_reg(self, ent_reg):
        """ Set the entropy regularization. """
        self._hyperparams['ent_reg'] = ent_reg

    def __getstate__(self):
        state = None
        return state

    def __setstate__(self, state):
        pass

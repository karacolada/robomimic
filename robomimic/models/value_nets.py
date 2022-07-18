"""
Contains torch Modules for value networks. These networks take an 
observation dictionary as input (and possibly additional conditioning, 
such as subgoal or goal dictionaries) and produce value or 
action-value estimates or distributions.
"""
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.obs_nets import MIMO_MLP, RNN_MIMO_MLP
from robomimic.models.distributions import DiscreteValueDistribution


class ValueNetwork(MIMO_MLP):
    """
    A basic value network that predicts values from observations.
    Can optionally be goal conditioned on future observations.
    """
    def __init__(
        self,
        obs_shapes,
        mlp_layer_dims,
        value_bounds=None,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for observations.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes. 

            value_bounds (tuple): a 2-tuple corresponding to the lowest and highest possible return
                that the network should be possible of generating. The network will rescale outputs
                using a tanh layer to lie within these bounds. If None, no tanh re-scaling is done.

            goal_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-observation key information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        self.value_bounds = value_bounds
        if self.value_bounds is not None:
            # convert [lb, ub] to a scale and offset for the tanh output, which is in [-1, 1]
            self._value_scale = (float(self.value_bounds[1]) - float(self.value_bounds[0])) / 2.
            self._value_offset = (float(self.value_bounds[1]) + float(self.value_bounds[0])) / 2.

        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes

        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        self._is_goal_conditioned = False
        if goal_shapes is not None and len(goal_shapes) > 0:
            assert isinstance(goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            self.goal_shapes = OrderedDict(goal_shapes)
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        else:
            self.goal_shapes = OrderedDict()

        output_shapes = self._get_output_shapes()
        super(ValueNetwork, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @MIMO_MLP, since we won't
        always directly predict values, but may instead predict the parameters
        of a value distribution.
        """
        return OrderedDict(value=(1,))

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [1]

    def forward(self, obs_dict, goal_dict=None):
        """
        Forward through value network, and then optionally use tanh scaling.
        """
        values = super(ValueNetwork, self).forward(obs=obs_dict, goal=goal_dict)["value"]
        if self.value_bounds is not None:
            values = self._value_offset + self._value_scale * torch.tanh(values)
        return values

    def _to_string(self):
        return "value_bounds={}".format(self.value_bounds)


class ActionValueNetwork(ValueNetwork):
    """
    A basic Q (action-value) network that predicts values from observations
    and actions. Can optionally be goal conditioned on future observations.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        value_bounds=None,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes. 

            value_bounds (tuple): a 2-tuple corresponding to the lowest and highest possible return
                that the network should be possible of generating. The network will rescale outputs
                using a tanh layer to lie within these bounds. If None, no tanh re-scaling is done.

            goal_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-observation key information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # add in action as a modality
        new_obs_shapes = OrderedDict(obs_shapes)
        new_obs_shapes["action"] = (ac_dim,)
        self.ac_dim = ac_dim

        # pass to super class to instantiate network
        super(ActionValueNetwork, self).__init__(
            obs_shapes=new_obs_shapes,
            mlp_layer_dims=mlp_layer_dims,
            value_bounds=value_bounds,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def forward(self, obs_dict, acts, goal_dict=None):
        """
        Modify forward from super class to include actions in inputs.
        """
        inputs = dict(obs_dict)
        inputs["action"] = acts
        return super(ActionValueNetwork, self).forward(inputs, goal_dict)

    def _to_string(self):
        return "action_dim={}\nvalue_bounds={}".format(self.ac_dim, self.value_bounds)


class DistributionalActionValueNetwork(ActionValueNetwork):
    """
    Distributional Q (action-value) network that outputs a categorical distribution over
    a discrete grid of value atoms. See https://arxiv.org/pdf/1707.06887.pdf for 
    more details.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        value_bounds,
        num_atoms,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes. 

            value_bounds (tuple): a 2-tuple corresponding to the lowest and highest possible return
                that the network should be possible of generating. This defines the support
                of the value distribution.

            num_atoms (int): number of value atoms to use for the categorical distribution - which
                is the representation of the value distribution.

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to DistributionalActionValueNetwork
        self.num_atoms = num_atoms
        self._atoms = np.linspace(value_bounds[0], value_bounds[1], num_atoms)

        # pass to super class to instantiate network
        super(DistributionalActionValueNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            mlp_layer_dims=mlp_layer_dims,
            value_bounds=value_bounds,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Network outputs log probabilities for categorical distribution over discrete value grid.
        """
        return OrderedDict(log_probs=(self.num_atoms,))

    def forward_train(self, obs_dict, acts, goal_dict=None):
        """
        Return full critic categorical distribution.

        Args:
            obs_dict (dict): batch of observations
            acts (torch.Tensor): batch of actions
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            value_distribution (DiscreteValueDistribution instance)
        """

        # add in actions
        inputs = dict(obs_dict)
        inputs["action"] = acts

        # network returns unnormalized log probabilities (logits) for each of the value atoms
        logits = MIMO_MLP.forward(self, obs=inputs, goal=goal_dict)["log_probs"]

        # turn these logits into a categorical distribution over the value atoms.
        # (unsqueeze to make sure atoms are compatible with batch operations)
        value_atoms = torch.Tensor(self._atoms).unsqueeze(0).to(logits.device)
        return DiscreteValueDistribution(values=value_atoms, logits=logits)

    def forward(self, obs_dict, acts, goal_dict=None):
        """
        Return mean of critic categorical distribution. Useful for obtaining
        point estimates of critic values.

        Args:
            obs_dict (dict): batch of observations
            acts (torch.Tensor): batch of actions
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            mean_value (torch.Tensor): expectation of value distribution
        """
        vd = self.forward_train(obs_dict=obs_dict, acts=acts, goal_dict=goal_dict)
        return vd.mean()

    def _to_string(self):
        return "action_dim={}\nvalue_bounds={}\nnum_atoms={}".format(self.ac_dim, self.value_bounds, self.num_atoms)

class RNNActionValueNetwork(RNN_MIMO_MLP):
    """
    A basic Q (action-value) network that predicts values from observations
    and actions. Can optionally be goal conditioned on future observations.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",
        rnn_kwargs=None,
        per_step=True,
        value_bounds=None,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes. 

            value_bounds (tuple): a 2-tuple corresponding to the lowest and highest possible return
                that the network should be possible of generating. The network will rescale outputs
                using a tanh layer to lie within these bounds. If None, no tanh re-scaling is done.

            goal_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-observation key information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        self.value_bounds = value_bounds
        if self.value_bounds is not None:
            # convert [lb, ub] to a scale and offset for the tanh output, which is in [-1, 1]
            self._value_scale = (float(self.value_bounds[1]) - float(self.value_bounds[0])) / 2.
            self._value_offset = (float(self.value_bounds[1]) + float(self.value_bounds[0])) / 2.

        assert isinstance(obs_shapes, OrderedDict)
        new_obs_shapes = OrderedDict(obs_shapes)
        new_obs_shapes["action"] = (ac_dim,)
        self.obs_shapes = new_obs_shapes
        self.ac_dim = ac_dim

        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        self._is_goal_conditioned = False
        if goal_shapes is not None and len(goal_shapes) > 0:
            assert isinstance(goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            self.goal_shapes = OrderedDict(goal_shapes)
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        else:
            self.goal_shapes = OrderedDict()

        output_shapes = self._get_output_shapes()

        super(RNNActionValueNetwork, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            mlp_layer_dims=mlp_layer_dims,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            rnn_kwargs=rnn_kwargs,
            per_step=per_step,
            encoder_kwargs=encoder_kwargs
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @MIMO_MLP, since we won't
        always directly predict values, but may instead predict the parameters
        of a value distribution.
        """
        return OrderedDict(value=(1,))

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [1]

    def forward(self, obs_dict, acts, goal_dict=None, rnn_init_state=None, return_state=False):
        """
        Forward through action value network, and then optionally use tanh scaling.
        """
        inputs = dict(obs_dict)
        inputs["action"] = acts

        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(goal_dict, size=obs_dict[mod].shape[1], dim=1)
        
        #values = super(RNNActionValueNetwork, self).forward(obs=inputs, goal=goal_dict)["value"]
        outputs = super(RNNActionValueNetwork, self).forward(obs=inputs, goal=goal_dict, rnn_init_state=rnn_init_state, return_state=return_state)
        
        if return_state:
            values, state = outputs
        else:
            values = outputs
            state = None
        
        if self.value_bounds is not None:
            values = self._value_offset + self._value_scale * torch.tanh(values)

        if return_state:
            return values, state
        else:
            return values

    def forward_step(self, obs_dict, acts, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get actions.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            acts: batch of actions
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            actions (torch.Tensor): batch of actions - does not contain time dimension
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        acts = TensorUtils.to_sequence(acts)
        value, state = self.forward(
            obs_dict, acts, goal_dict, rnn_init_state=rnn_state, return_state=True)
        return value[:, 0], state        

    def _to_string(self):
        return "action_dim={}\nvalue_bounds={}".format(self.ac_dim, self.value_bounds)

"""Temporal Fusion Transformer Model.

Contains the full TFT architecture and associated components. Defines functions
for training, evaluation and prediction using simple Pandas Dataframe inputs.
"""

import data_formatters.base
import libs.utils as utils
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import copy


# Default input types.
InputTypes = data_formatters.base.InputTypes


# Implementation of linear = tf.keras.layers.TimeDistributed(linear) with pytorch
class TimeDistributed(nn.Module):
    def __init__(self, layer, batch_first=False):
        super(TimeDistributed, self).__init__()
        if not isinstance(layer, nn.Module):
            raise ValueError(
                "Please initialize `TimeDistributed` layer with a "
                f"`nn.Module` instance. Received: {layer}"
            )
        self.layer = layer
        self.batch_first = batch_first
    
    def forward(self, x):
        if len(x.size()) <= 2:
            return self.layer(x)
        # Reshape input tensor to (batch_size*seq_len, input_size)
        batch_size, seq_len, input_size = x.size()
        x_reshaped = x.view(-1, input_size)
        # Apply the layer to the reshaped input
        output_reshaped = self.layer(x_reshaped)
        # Reshape the output back to (batch_size, seq_len, output_size)
        if self.batch_first:
            output = output_reshaped.view(batch_size, seq_len, -1)
        else:
            output = output_reshaped.view(seq_len, batch_size, -1)
        return output


class GLU(nn.Module):
    """Gated Linear Units.

    Args:
        input_dim: int
            The embedding size of the input.

    Returns:
        A tensor with the same shape as input_dim.
    """
    def __init__(self, input_dim: int):
        super(GLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc2 = nn.Linear(input_dim, input_dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.fc2(x)
        x = self.fc2(x)
        return torch.mul(sig, x)


class GRN(nn.Module):
    """Gated Residual Network.

    Args:
        input_dim: int
            The embedding size of the input.
        hidden_dim: int
            The intermediate embedding size.
        output_dim: int
            The embedding size of the output.
        dropout: Optional[float]
            The dropout rate associated with the component.
        context_dim: Optional[int]
            The embedding size of the context signal.
        batch_first: Optional[bool]
            A boolean indicating whether the batch dimension is expected to be the first dimension of the output.

    Returns:
        A tensor with the same shape as output_dim which can be not the same as input_dim.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: Optional[float] = 0.05,
                 context_dim: Optional[int] = None,
                 batch_first: Optional[bool] = True):
        super(GRN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.context_dim = context_dim

        self.project_residual: bool = self.input_dim != self.output_dim
        if self.project_residual:
            self.skip_layer = TimeDistributed(nn.Linear(self.input_dim, self.hidden_dim, bias=True), batch_first=batch_first)
        
        self.fc1 = TimeDistributed(nn.Linear(self.input_dim, self.hidden_dim, bias=True), batch_first=batch_first)

        if self.context_dim is not None:
            self.context_projection = TimeDistributed(nn.Linear(self.context_dim, self.hidden_dim, bias=False), batch_first=batch_first)
        
        self.elu = nn.ELU()

        self.fc2 = TimeDistributed(nn.Linear(self.hidden_dim, self.output_dim, bias=True), batch_first=batch_first)

        self.dropout = nn.Dropout(self.dropout)
        self.gate = TimeDistributed(GLU(self.output_dim), batch_first=batch_first)
        self.layernorm = TimeDistributed(nn.LayerNorm(self.output_dim), batch_first=batch_first)

    def forward(self, x, context=None):
        if self.project_residual:
            residual = self.skip_layer(x)
        else:
            residual = x

        x = self.fc1(x)
        if context is not None:
            context = self.context_projection(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = x + residual
        x = self.layernorm(x)

        return x
    

class VariableSelectionNetwork(nn.Module):
    """
    Args:
        input_dim: int
            The embedding size of the input, associated with the state_size of the model.
        num_inputs: int
            The quantity of input variables, including both numeric and categorical inputs for the relevant channel.
        hidden_dim: int
            The embedding size of the output.
        dropout: float
            The dropout rate associated with GRN composing this object.
        context_dim: Optional[int]
            The embedding size of the context signal.
        batch_first: Optional[bool]
            A boolean indicating whether the batch dimension is expected to be the first dimension of the output.
    
    Returns:
        outputs: [(num_samples * num_temporal_steps) x state_size]
        sparse_weights: [(num_samples * num_temporal_steps) x num_inputs x 1] # weights Vxt
    """

    def __init__(self, input_dim: int, num_inputs: int, hidden_dim: int, dropout: float,
                 context_dim: Optional[int] = None, 
                 batch_first: Optional[bool] = True):
        super(VariableSelectionNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.context_dim = context_dim

        self.flattened_grn = GRN(input_dim=self.num_inputs * self.input_dim,
                                 hidden_dim=self.hidden_dim,
                                 output_dim=self.num_inputs,
                                 dropout=self.dropout,
                                 context_dim=self.context_dim,
                                 batch_first=batch_first)
        self.softmax = nn.Softmax(dim=1)

        self.single_variable_grns = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.single_variable_grns.append(
                GRN(input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    output_dim=self.hidden_dim,
                    dropout=self.dropout,
                    batch_first=batch_first))
    
    def forward(self, flattened_embedding, context=None):
        # the shape of flattened embedding should be [(num_samples * num_temporal_steps) x (num_inputs x input_dim)]
        sparse_weights = self.flattened_grn(flattened_embedding, context)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
        # After this step, the shape of sparse_weights is [(num_samples * num_temporal_steps) x num_inputs x 1]

        processed_inputs = []
        for i in range(self.num_inputs):
            processed_inputs.append(
                self.single_variable_grns[i](flattened_embedding[..., (i * self.input_dim): (i + 1) * self.input_dim]))
        # Each element in the resulting list is of size: [(num_samples * num_temporal_steps) x state_size],
        # and each element corresponds to a single input variable

        # Combine the outputs of the single-var GRNs (along an additional axis)
        processed_inputs = torch.stack(processed_inputs, dim=-1)
        # processed_inputs: [(num_samples * num_temporal_steps) x state_size x num_inputs]

        outputs = processed_inputs * sparse_weights.transpose(1, 2)
        # outputs: [(num_samples * num_temporal_steps) x state_size x num_inputs]
        outputs = outputs.sum(axis=-1)
        # outputs: [(num_samples * num_temporal_steps) x state_size]
        return outputs, sparse_weights
    

class NumericInputTransformation(nn.Module):
    """
    Args:
        num_inputs : int
            The quantity of numeric input variables associated with this module.
        state_size : int
            The state size of the model, which determines the embedding dimension.

    Returns:
        projections:
    """
    def __init__(self, num_inputs: int, state_size: int):
        super(NumericInputTransformation, self).__init__()
        self.num_inputs = num_inputs
        self.state_size = state_size
        self.numeric_projection_layers = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.numeric_projection_layers.append(nn.Linear(1, self.state_size))
        
    def forward(self, x: torch.tensor) -> List[torch.tensor]:
        projections = []
        for i in range(self.num_inputs):
            projections.append(self.numeric_projection_layers[i](x[:, [i]]))
        return projections
    

class CategoricalInputTransformation(nn.Module):
    """
    Args:
        num_inputs : int
            The quantity of categorical input variables associated with this module.
        state_size : int
            The state size of the model, which determines the embedding dimension.
        cardinalities: List[int]
            The quantity of categories associated with each of the input variables.
    
    Returns:
        embeddings:
    """
    def __init__(self, num_inputs: int, state_size: int, cardinalities: List[int]):
        super(CategoricalInputTransformation, self).__init__()
        self.num_inputs = num_inputs
        self.state_size = state_size
        self.cardinalities = cardinalities
        self.categorical_embedding_layers = nn.ModuleList()
        for idx, cardinality in enumerate(self.cardinalities):
            self.categorical_embedding_layers.append(nn.Embedding(cardinality, self.state_size))

    def forward(self, x: torch.tensor) -> List[torch.tensor]:
        embeddings = []
        for i in range(self.num_inputs):
            embeddings.append(self.categorical_embedding_layers[i](x[:, i]))
        return embeddings


class InputChannelEmbedding(nn.Module):
    """
    Args:
        state_size: int
        num_numeric: int
            The quantity of numeric input variables.
        num_categorical: int
            The quantity of categorical input variables.
        categorical_cardinalities: List[int]
            The quantity of categories associated with each of the categorical input variables.
        time_distribute: Optional[bool]
    Returns:
        merged_transformations:
    """
    def __init__(self, state_size: int, num_numeric: int, num_categorical: int, categorical_cardinalities: List[int],
                 time_distribute: Optional[bool] = False):
        super(InputChannelEmbedding, self).__init__()
        self.state_size = state_size
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical
        self.categorical_cardinalities = categorical_cardinalities
        self.time_distribute = time_distribute

        if (num_numeric + num_categorical) < 1:
            raise ValueError(f"""At least a single input variable (either numeric or categorical) should be included
            as part of the input channel.
            According to the provided configuration:
            num_numeric + num_categorical = {num_numeric} + {num_categorical} = {num_numeric + num_categorical} < 1
            """)
        
        if self.time_distribute:
            self.numeric_transform = TimeDistributed(
                NumericInputTransformation(num_inputs=num_numeric, state_size=state_size), batch_first=True)
            self.categorical_transform = TimeDistributed(
                CategoricalInputTransformation(num_inputs=num_categorical, state_size=state_size,
                                               cardinalities=categorical_cardinalities), batch_first=True)
        else:    
            self.numeric_transform = NumericInputTransformation(num_inputs=num_numeric, state_size=state_size)
            self.categorical_transform = CategoricalInputTransformation(num_inputs=num_categorical,
                                                                        state_size=state_size,
                                                                        cardinalities=categorical_cardinalities)
        
        if num_numeric == 0:
            self.numeric_transform = []
        if num_categorical == 0:
            self.categorical_transform = []

    def forward(self, x_numeric, x_categorical) -> torch.tensor:
        batch_shape = x_numeric.shape if x_numeric.nelement()>0 else x_categorical.shape
        processed_numeric = self.numeric_transform(x_numeric)
        processed_categorical = self.categorical_transform(x_categorical)
        # [num_samples × num_temporal_steps x state_size]
        # (for the static input channel, num_temporal_steps is irrelevant and can be treated as 1
        merged_transformations = torch.cat(processed_numeric + processed_categorical, dim=2)
        # merged_transformations: [num_samples x num_temporal_steps x (state_size * total_input_variables)]
        return merged_transformations
    

class GateAddNorm(nn.Module):
    """
    This composite operation includes:
    a. A *Dropout* layer.
    b. Gating using a ``GatedLinearUnit``.
    c. A residual connection to an "earlier" signal from the forward pass of the parent model.
    d. Layer normalization.

    Args:
        input_dim: int
        dropout: Optional[float]
    Returns:
        x: torch.tensor
    """
    def __init__(self, input_dim: int, dropout: Optional[float] = None):
        super(GateAddNorm, self).__init__()
        self.dropout_rate = dropout
        if dropout:
            self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.gate = TimeDistributed(GLU(input_dim), batch_first=True)
        self.layernorm = TimeDistributed(nn.Linear(input_dim), batch_first=True)

    def forward(self, x, residual=None):
        if self.dropout_rate:
            x = self.dropout_layer(x)
        x = self.gate(x)
        if residual is not None:
            x = x + residual
        x = self.layernorm(x)


class InterpretableMultiHeadAttention(nn.Module):
    """
    Args:
        embed_dim: int
        num_heads: int
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.d_model = embed_dim
        self.num_heads = num_heads
        self.all_heads_dim = embed_dim * num_heads
        self.w_q = nn.Linear(embed_dim, self.all_heads_dim) # the default value of 'bias' is True
        self.w_k = nn.Linear(embed_dim, self.all_heads_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(self.d_model, self.d_model)

    def attention(self, q, k, v, mask=None):
        # Scale dot product.
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        # Shape: [num_samples x num_heads x num_future_steps x num_total_steps]
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, -1e9)

        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
        # Shpae: [num_samples x num_heads x num_future_steps x num_total_steps]
        attention_outputs = torch.matmul(attention_scores, v)
        # Shpae: [num_samples x num_heads x num_future_steps x state_size]
        return attention_outputs, attention_scores
    
    def forward(self, q, k, v, mask=None):
        num_samples = q.size(0)
        # q: [num_samples x num_future_steps x state_size]
        # k: [num_samples x (num_total_steps) x state_size]
        # v: [num_samples x (num_total_steps) x state_size]
        q_proj = self.w_q(q).view(num_samples, -1, self.num_heads, self.d_model)
        k_proj = self.w_k(k).view(num_samples, -1, self.num_heads, self.d_model)
        v_proj = self.w_v(v).repeat(1, 1, self.num_heads).view(num_samples, -1, self.num_heads, self.d_model)
        q_proj = q_proj.transpose(1, 2)  # (num_samples x num_future_steps x num_heads x state_size)
        k_proj = k_proj.transpose(1, 2)  # (num_samples x num_total_steps x num_heads x state_size)
        v_proj = v_proj.transpose(1, 2)  # (num_samples x num_total_steps x num_heads x state_size)
        attention_outputs_all_heads, attention_scores_all_heads = self.attention(q_proj, k_proj, v_proj, mask)
        # attention_scores_all_heads: [num_samples x num_heads x num_future_steps x num_total_steps]
        # attention_outputs_all_heads: [num_samples x num_heads x num_future_steps x state_size]
        attention_outpus = attention_outputs_all_heads.mean(dim=1)
        attention_scores = attention_scores_all_heads.mean(dim=1)
        # attention_outputs: [num_samples x num_future_steps x state_size]
        # attention_scores: [num_samples x num_future_steps x num_total_steps]
        output = self.out(attention_outpus)
        # output: [num_samples x num_future_steps x state_size]
        return output, attention_outpus, attention_scores


class TFT(nn.Module):
    """
    Args:
        config: Dict
            A mapping describing both the expected structure of the input of the model, and the architectural specification
            of the model.
            This mapping should include a key named ``data_props`` in which the dimensions and cardinalities (where the
            inputs are categorical) are specified. Moreover, the configuration mapping should contain a key named ``model``,
            specifying ``attention_heads`` , ``dropout`` , ``lstm_layers`` , ``output_quantiles`` and ``state_size`` ,
            which are required for creating the model.
            config = {
                "data_props": {
                    "num_historical_numeric": 10,
                    "num_historical_categorical": 10,
                    "historical_categorical_cardinalities": [5, 10, 3],
                },
                "model": {
                    "attention_heads": 4,
                    "dropout": 0.2,
                    "lstm_layers": 2,
                    "output_quantiles": [0.1, 0.5, 0.9],
                    "state_size": 256
                }
                "task_type": 'regression'
                "target_window_start": None
            }
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # ============
        # data props
        # ============
        data_props = config['data_props']
        self.num_historical_numeric = data_props['num_historical_numeric']
        self.num_historical_categorical = data_props['num_historical_categorical']
        self.historical_categorical_cardinalities = data_props['historical_categorical_cardinalities']
        self.num_static_numeric = data_props['num_static_numeric']
        self.num_static_categorical = data_props['num_static_categorical']
        self.static_categorical_cardinalities = data_props['static_categorical_cardinalities']
        self.num_future_numeric = data_props['num_future_numeric']
        self.num_future_categorical = data_props['num_future_categorical']
        self.future_categorical_cardinalities = data_props['future_categorical_cardinalities']

        self.historical_ts_representative_key = 'historical_ts_numeric' if self.num_historical_numeric > 0 \
            else 'historical_ts_categorical'
        self.future_ts_representative_key = 'future_ts_numeric' if self.num_future_numeric > 0 \
            else 'future_ts_categorical'
        
        # ============
        # model props
        # ============
        self.task_type = config['task_type']
        model = config['model']
        self.attention_heads = model['attention_heads']
        self.dropout = model['dropout']
        self.lstm_layers = model['lstm_layers']
        self.target_window_start_idx = (config['target_window_start'] - 1) if config['target_window_start'] is not None else 0
        if self.task_type == 'regression':
            self.output_quantiles = model['output_quantiles']
            self.num_outputs = len(self.output_quantiles)
        elif self.task_type == 'classification':
            self.output_quantiles = None
            self.num_outputs = 1
        else:
            raise ValueError(f"unsupported task type: {self.task_type}")
        self.state_size = model['state_size']

        # =====================
        # Input Transformation
        # =====================
        self.static_transform = InputChannelEmbedding(state_size=self.state_size,
                                                      num_numeric=self.num_static_numeric,
                                                      num_categorical=self.num_static_categorical,
                                                      categorical_cardinalities=self.static_categorical_cardinalities,
                                                      time_distribute=False)
        self.historical_ts_transform = InputChannelEmbedding(
            state_size=self.state_size,
            num_numeric=self.num_historical_numeric,
            num_categorical=self.num_historical_categorical,
            categorical_cardinalities=self.historical_categorical_cardinalities,
            time_distribute=True)
        self.future_ts_transform = InputChannelEmbedding(
            state_size=self.state_size,
            num_numeric=self.num_future_numeric,
            num_categorical=self.num_future_categorical,
            categorical_cardinalities=self.future_categorical_cardinalities,
            time_distribute=True)
        
        # =============================
        # Variable Selection Networks
        # =============================
        self.static_selection = VariableSelectionNetwork(
            input_dim=self.state_size,
            num_inputs=self.num_static_numeric + self.num_static_categorical,
            hidden_dim=self.state_size,
            dropout=self.dropout)
        self.historical_ts_selection = VariableSelectionNetwork(
            input_dim=self.state_size,
            num_inputs=self.num_historical_numeric + self.num_historical_categorical,
            hidden_dim=self.state_size,
            dropout=self.dropout,
            context_dim=self.state_size)
        self.future_ts_selection = VariableSelectionNetwork(
            input_dim=self.state_size,
            num_inputs=self.num_future_numeric + self.num_future_categorical,
            hidden_dim=self.state_size,
            dropout=self.dropout,
            context_dim=self.state_size)
        
        # =============================
        # static covariate encoders
        # =============================
        static_covariate_encoder = GRN(input_dim=self.state_size,
                                       hidden_dim=self.state_size,
                                       output_dim=self.state_size,
                                       dropout=self.dropout)
        self.static_encoder_selection = copy.deepcopy(static_covariate_encoder)
        self.static_encoder_environment = copy.deepcopy(static_covariate_encoder)
        self.static_encoder_sequential_cell_init = copy.deepcopy(static_covariate_encoder)
        self.static_encoder_sequential_state_init = copy.deepcopy(static_covariate_encoder)

        # ============================================================
        # Locality Enhancement with Sequence-to-Sequence processing
        # ============================================================
        self.past_lstm = nn.LSTM(input_size=self.state_size,
                                 hidden_size=self.state_size,
                                 num_layers=self.lstm_layers,
                                 dropout=self.dropout,
                                 batch_first=True)
        self.future_lstm = nn.LSTM(input_size=self.state_size,
                                   hidden_size=self.state_size,
                                   num_layers=self.lstm_layers,
                                   dropout=self.dropout,
                                   batch_first=True)
        self.post_lstm_gating = GateAddNorm(input_dim=self.state_size, dropout=self.dropout)

        # ============================================================
        # Static enrichment
        # ============================================================
        self.static_enrichment_grn = GRN(input_dim=self.state_size,
                                         hidden_dim=self.state_size,
                                         output_dim=self.state_size,
                                         context_dim=self.state_size,
                                         dropout=self.dropout)
        
        # ============================================================
        # Temporal Self-Attention
        # ============================================================
        self.multihead_attention = InterpretableMultiHeadAttention(embed_dim=self.state_size, num_heads=self.attention_heads)
        self.post_attention_gating = GateAddNorm(input_dim=self.state_size, dropout=None)

        # ============================================================
        # Position-wise feed forward
        # ============================================================
        self.pos_wise_ff_grn = GRN(input_dim=self.state_size,
                                                    hidden_dim=self.state_size,
                                                    output_dim=self.state_size,
                                                    dropout=self.dropout)
        self.pos_wise_ff_gating = GateAddNorm(input_dim=self.state_size, dropout=None)

        # ============================================================
        # Output layer
        # ============================================================
        self.output_layer = nn.Linear(self.state_size, self.num_outputs)

    @staticmethod
    def replicate_along_time(static_signal: torch.tensor, time_steps: int) -> torch.tensor:
        """
        This method gets as an input a static_signal (non-temporal tensor) [num_samples x num_features],
        and replicates it along time for 'time_steps' times,
        creating a tensor of [num_samples x time_steps x num_features]

        Args:
            static_signal: the non-temporal tensor for which the replication is required.
            time_steps: the number of time steps according to which the replication is required.
        Returns:
            torch.tensor: the time-wise replicated tensor.
        """
        time_distributed_signal = static_signal.unsqueeze(1).repeat(1, time_steps, 1)
        return time_distributed_signal
    
    @staticmethod
    def stack_time_steps_along_batch(temporal_signal: torch.tensor) -> torch.tensor:
        """
        This method gets as an input a temporal signal [num_samples x time_steps x num_features]
        and stacks the batch dimension and the temporal dimension on the same axis (dim=0).

        The last dimension (features dimension) is kept, but the rest is stacked along dim=0.
        """
        return temporal_signal.view(-1, temporal_signal.size(-1))

    def apply_temporal_selection(self, temporal_representation: torch.tensor,
                                 static_selection_signal: torch.tensor,
                                 temporal_selection_module:VariableSelectionNetwork
                                 ) -> Tuple[torch.tensor, torch.tensor]:
        num_samples, num_temporal_steps, _ = temporal_representation.shape
        time_distributed_context = self.replicate_along_time(static_signal=static_selection_signal,
                                                             time_steps=num_temporal_steps)
        # time_distributed_context: [num_samples x num_temporal_steps x state_size]
        # temporal_representation: [num_samples x num_temporal_steps x (total_num_temporal_inputs * state_size)]
        temporal_flattened_embedding = self.stack_time_steps_along_batch(temporal_representation)
        time_distributed_context = self.stack_time_steps_along_batch(time_distributed_context)
        # temporal_flattened_embedding: [(num_samples * num_temporal_steps) x (total_num_temporal_inputs * state_size)]
        # time_distributed_context: [(num_samples * num_temporal_steps) x state_size]
        temporal_selection_output, temporal_selection_weights = temporal_selection_module(
            flattened_embedding=temporal_flattened_embedding, context=time_distributed_context)
        # temporal_selection_output: [(num_samples * num_temporal_steps) x state_size]
        # temporal_selection_weights: [(num_samples * num_temporal_steps) x (num_temporal_inputs) x 1]
        temporal_selection_output = temporal_selection_output.view(num_samples, num_temporal_steps, -1)
        temporal_selection_weights = temporal_selection_weights.squeeze(-1).view(num_samples, num_temporal_steps, -1)
        # temporal_selection_output: [num_samples x num_temporal_steps x state_size)]
        # temporal_selection_weights: [num_samples x num_temporal_steps x num_temporal_inputs)]
        return temporal_selection_output, temporal_selection_weights
    
    def transform_inputs(self, batch: Dict[str, torch.tensor]) -> Tuple[torch.tensor, ...]:
        """
        This method processes the batch and transform each input channel (historical_ts, future_ts, static)
        separately to eventually return the learned embedding for each of the input channels

        each feature is embedded to a vector of state_size dimension:
        - numeric features will be projected using a linear layer
        - categorical features will be embedded using an embedding layer

        eventually the embedding for all the features will be concatenated together on the last dimension of the tensor
        (i.e. dim=1 for the static features, dim=2 for the temporal data).
        """
        empty_tensor = torch.empty((0, 0))
        static_rep = self.static_transform(x_numeric=batch['static_feats_numeric'],
                                           x_categorical=batch['static_feats_categorical'])
        historical_ts_rep = self.historical_ts_transform(x_numeric=batch['historical_ts_numeric'],
                                                         x_categorical=batch['historical_ts_categorical'])
        future_ts_rep = self.future_ts_transform(x_numeric=batch['future_ts_numeric'],
                                                 x_categorical=batch['future_ts_categorical'])
        return future_ts_rep, historical_ts_rep, static_rep
    
    def get_static_encoders(self, selected_static: torch.tensor) -> Tuple[torch.tensor, ...]:
        """
        This method processes the variable selection results for the static data, yielding signals which are designed
        to allow better integration of the information from static metadata.
        Each of the resulting signals is generated using a separate GRN, and is eventually wired into various locations
        in the temporal fusion decoder, for allowing static variables to play an important role in processing.

        c_selection will be used for temporal variable selection
        c_seq_hidden & c_seq_cell will be used both for local processing of temporal features
        c_enrichment will be used for enriching temporal features with static information.
        """
        c_selection = self.static_encoder_selection(selected_static)
        c_enrichment = self.static_encoder_environment(selected_static)
        c_seq_hidden = self.static_encoder_sequential_state_init(selected_static)
        c_seq_cell = self.static_encoder_sequential_cell_init(selected_static)
        return c_enrichment, c_selection, c_seq_cell, c_seq_hidden
    
    def apply_sequential_processing(self, selected_historical: torch.tensor, selected_future: torch.tensor,
                                    c_seq_hidden: torch.tensor, c_seq_cell: torch.tensor) -> torch.tensor:
        """
        This part of the model is designated to mimic a sequence-to-sequence layer which will be used for local
        processing.
        On that part the historical ("observed") information will be fed into a recurrent layer called "Encoder" and
        the future information ("known") will be fed into a recurrent layer called "Decoder".
        This will generate a set of uniform temporal features which will serve as inputs into the temporal fusion
        decoder itself.
        To allow static metadata to influence local processing, we use "c_seq_hidden" and "c_seq_cell" context vectors
        from the static covariate encoders to initialize the hidden state and the cell state respectively.
        The output of the recurrent layers is gated and fused with a residual connection to the input of this block.
        """
        # concatenate the observed temporal signal with the known temporal singal, along the time dimension
        lstm_input = torch.cat([selected_historical, selected_future], dim=1)
        past_lstm_output, hidden = self.past_lstm(selected_historical,
                                                  (c_seq_hidden.unsqueeze(0).repeat(self.lstm_layers, 1, 1),
                                                   c_seq_cell.unsqueeze(0).repeat(self.lstm_layers, 1, 1)))
        future_lstm_output, _ = self.future_lstm(selected_future, hidden)
        lstm_output = torch.cat([past_lstm_output, future_lstm_output])
        gated_lstm_output = self.post_lstm_gating(lstm_output, residual=lstm_input)
        return gated_lstm_output
    
    def apply_static_enrichment(self, gated_lstm_output: torch.tensor,
                                static_enrichment_signal: torch.tensor) -> torch.tensor:
        """
        This static enrichment stage enhances temporal features with static metadata using a GRN.
        The static enrichment signal is an output of a static covariate encoder, and the GRN is shared across time.
        """
        num_samples, num_temporal_steps, _ = gated_lstm_output.shape
        time_distributed_context = self.replicate_along_time(static_signal=static_enrichment_signal,
                                                             time_steps=num_temporal_steps)
        # time_distributed_context: [num_samples x num_temporal_steps x state_size]
        flattened_gated_lstm_output = self.stack_time_steps_along_batch(gated_lstm_output)
        time_distributed_context = self.stack_time_steps_along_batch(time_distributed_context)
        # flattened_gated_lstm_output: [(num_samples * num_temporal_steps) x state_size]
        # time_distributed_context: [(num_samples * num_temporal_steps) x state_size]
        enriched_sequence = self.static_enrichment_grn(flattened_gated_lstm_output,
                                                       context=time_distributed_context)
        # enriched_sequence: [(num_samples * num_temporal_steps) x state_size]
        enriched_sequence = enriched_sequence.view(num_samples, -1, self.state_size)
        # enriched_sequence: [num_samples x num_temporal_steps x state_size]
        return enriched_sequence
    
    def apply_self_attention(self, enriched_sequence: torch.tensor,
                             num_historical_steps: int,
                             num_future_steps: int):
        # create a mask - so that future steps will be exposed (able to attend) only to preceding steps
        output_sequence_length = num_future_steps - self.target_window_start_idx
        mask = torch.cat([torch.zeros(output_sequence_length,
                                      num_historical_steps + self.target_window_start_idx,
                                      device=enriched_sequence.device),
                          torch.triu(torch.ones(output_sequence_length, output_sequence_length,
                                                device=enriched_sequence.device),
                                     diagonal=1)], dim=1)
        # mask: [output_sequence_length x (num_historical_steps + num_future_steps)]
        post_attention, attention_outputs, attention_scores = self.multihead_attention(
            q=enriched_sequence[:, (num_historical_steps + self.target_window_start_idx):, :],
            k=enriched_sequence,
            v=enriched_sequence,
            mask=mask.bool())
        # post_attention: [num_samples x num_future_steps x state_size]
        # attention_outputs: [num_samples x num_future_steps x state_size]
        # attention_scores: [num_samples x num_future_steps x num_total_steps]
        gated_post_attention = self.post_attention_gating(
            x=post_attention,
            residual=enriched_sequence[:, (num_historical_steps + self.target_window_start_idx):, :])
        # gated_post_attention: [num_samples x num_future_steps x state_size]
        return gated_post_attention, attention_scores
    
    def forward(self, batch):
        """
        batch = {
            'static_feats_numeric': torch.Tensor,  # 静态数值特征，形状：[num_samples x num_static_numeric]
            'static_feats_categorical': torch.Tensor,  # 静态分类特征，形状：[num_samples x num_static_categorical]
            'historical_ts_numeric': torch.Tensor,  # 历史数值时间序列，形状：[num_samples x num_historical_steps x num_historical_numeric]
            'historical_ts_categorical': torch.Tensor,  # 历史分类时间序列，形状：[num_samples x num_historical_steps x num_historical_categorical]
            'future_ts_numeric': torch.Tensor,  # 未来数值时间序列，形状：[num_samples x num_future_steps x num_future_numeric]
            'future_ts_categorical': torch.Tensor,  # 未来分类时间序列，形状：[num_samples x num_future_steps x num_future_categorical],
            # 可以包含其他模型需要的信息，例如 'target_window_start': int
        }
        """
        num_samples, num_historical_steps, _ = batch[self.historical_ts_representative_key].shape
        num_future_steps = batch[self.future_ts_representative_key].shape[1]
        # define output_sequence_length : num_future_steps - self.target_window_start_idx

        # =========== Transform all input channels ==============
        future_ts_rep, historical_ts_rep, static_rep = self.transform_inputs(batch)
        # static_rep: [num_samples x (total_num_static_inputs * state_size)]
        # historical_ts_rep: [num_samples x num_historical_steps x (total_num_historical_inputs * state_size)]
        # future_ts_rep: [num_samples x num_future_steps x (total_num_future_inputs * state_size)]

        # =========== Static Variables Selection ==============
        selected_static, static_weights = self.static_selection(static_rep)
        # selected_static: [num_samples x state_size]
        # static_weights: [num_samples x num_static_inputs x 1]

        # =========== Static Covariate Encoding ==============
        c_enrichment, c_selection, c_seq_cell, c_seq_hidden = self.get_static_encoders(selected_static)
        # each of the static encoders signals is of shape: [num_samples x state_size]

        # =========== Historical variables selection ==============
        selected_historical, historical_selection_weights = self.apply_temporal_selection(
            temporal_representation=historical_ts_rep,
            static_selection_signal=c_selection,
            temporal_selection_module=self.historical_ts_selection)
        # selected_historical: [num_samples x num_historical_steps x state_size]
        # historical_selection_weights: [num_samples x num_historical_steps x total_num_historical_inputs]

        # =========== Future variables selection ==============
        selected_future, future_selection_weights = self.apply_temporal_selection(
            temporal_representation=future_ts_rep,
            static_selection_signal=c_selection,
            temporal_selection_module=self.future_ts_selection)
        # selected_future: [num_samples x num_future_steps x state_size]
        # future_selection_weights: [num_samples x num_future_steps x total_num_future_inputs]

        # =========== Locality Enhancement - Sequential Processing ==============
        gated_lstm_output = self.apply_sequential_processing(selected_historical=selected_historical,
                                                             selected_future=selected_future,
                                                             c_seq_hidden=c_seq_hidden,
                                                             c_seq_cell=c_seq_cell)
        # gated_lstm_output : [num_samples x (num_historical_steps + num_future_steps) x state_size]

        # =========== Static enrichment ==============
        enriched_sequence = self.apply_static_enrichment(gated_lstm_output=gated_lstm_output,
                                                         static_enrichment_signal=c_enrichment)
        # enriched_sequence: [num_samples x (num_historical_steps + num_future_steps) x state_size]

        # =========== self-attention ==============
        gated_post_attention, attention_scores = self.apply_self_attention(enriched_sequence=enriched_sequence,
                                                                           num_historical_steps=num_historical_steps,
                                                                           num_future_steps=num_future_steps)
        # attention_scores: [num_samples x output_sequence_length x (num_historical_steps + num_future_steps)]
        # gated_post_attention: [num_samples x output_sequence_length x state_size]

        # =========== position-wise feed-forward ==============
        post_poswise_ff_grn = self.pos_wise_ff_grn(gated_post_attention)
        gated_poswise_ff = self.pos_wise_ff_gating(
            post_poswise_ff_grn,
            residual=gated_lstm_output[:, (num_historical_steps + self.target_window_start_idx):, :])
        # gated_poswise_ff: [num_samples x output_sequence_length x state_size]

        # =========== output projection ==============
        predicted_quantiles = self.output_layer(gated_poswise_ff)
        # predicted_quantiles: [num_samples x num_future_steps x num_quantiles]

        return {
            'predicted_quantiles': predicted_quantiles,  # [num_samples x output_sequence_length x num_quantiles]
            'static_weights': static_weights.squeeze(-1),  # [num_samples x num_static_inputs]
            'historical_selection_weights': historical_selection_weights,
            # [num_samples x num_historical_steps x total_num_historical_inputs]
            'future_selection_weights': future_selection_weights,
            # [num_samples x num_future_steps x total_num_future_inputs]
            'attention_scores': attention_scores
            # [num_samples x output_sequence_length x (num_historical_steps + num_future_steps)]
        }
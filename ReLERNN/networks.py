"""
Authors: Jeff Adrion, Andrew Kern, Jared Galloway, Silas Tittes
"""

from ReLERNN.imports import *


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
    input_shape,
    position_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):

    inputs = input_shape
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return Model([inputs, position_shape], outputs)


def TRANSFORMER_PILOT(x, y):
    """
    Verifying updates, but will eventually be transformer model.
    """

    haps, pos = x

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    genotype_inputs = layers.Input(shape=(numSNPs, numSamps))

    position_inputs = layers.Input(shape=(numPos,))

    model = build_model(
        input_shape=genotype_inputs,
        position_shape=position_inputs,
        head_size=128,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    model.compile(
        optimizer="Adam",
        loss="mse",
    )
    model.summary()

    return model


def GRU_TUNED84(x, y):
    """
    Same as GRU_VANILLA but with dropout AFTER each dense layer.
    """

    haps, pos = x

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    genotype_inputs = layers.Input(shape=(numSNPs, numSamps))
    model = layers.Bidirectional(layers.GRU(84, return_sequences=False))(
        genotype_inputs
    )
    model = layers.Dense(256)(model)
    model = layers.Dropout(0.35)(model)

    # ----------------------------------------------------

    position_inputs = layers.Input(shape=(numPos,))
    m2 = layers.Dense(256)(position_inputs)

    # ----------------------------------------------------

    model = layers.concatenate([model, m2])
    model = layers.Dense(64)(model)
    model = layers.Dropout(0.35)(model)
    output = layers.Dense(1)(model)

    # ----------------------------------------------------

    model = Model(inputs=[genotype_inputs, position_inputs], outputs=[output])
    model.compile(optimizer="Adam", loss="mse")
    model.summary()

    return model


def GRU_POOLED(x, y):

    sites = x.shape[1]
    features = x.shape[2]

    genotype_inputs = layers.Input(shape=(sites, features))
    model = layers.Bidirectional(layers.GRU(84, return_sequences=False))(
        genotype_inputs
    )
    model = layers.Dense(256)(model)
    model = layers.Dropout(0.35)(model)
    output = layers.Dense(1)(model)

    model = Model(inputs=[genotype_inputs], outputs=[output])
    model.compile(optimizer="Adam", loss="mse")
    model.summary()

    return model


def HOTSPOT_CLASSIFY(x, y):

    haps, pos = x

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    genotype_inputs = layers.Input(shape=(numSNPs, numSamps))
    model = layers.Bidirectional(layers.GRU(84, return_sequences=False))(
        genotype_inputs
    )
    model = layers.Dense(256)(model)
    model = layers.Dropout(0.35)(model)

    # ----------------------------------------------------

    position_inputs = layers.Input(shape=(numPos,))
    m2 = layers.Dense(256)(position_inputs)

    # ----------------------------------------------------

    model = layers.concatenate([model, m2])
    model = layers.Dense(64)(model)
    model = layers.Dropout(0.35)(model)
    output = layers.Dense(1, activation="sigmoid")(model)

    # ----------------------------------------------------

    model = Model(inputs=[genotype_inputs, position_inputs], outputs=[output])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.summary()

    return model


NetworkDictionary = {
    "TRANSFORMER_PILOT": TRANSFORMER_PILOT,
    "GRU_TUNED84": GRU_TUNED84,
    "GRU_POOLED": GRU_POOLED,
    "HOTSPOT_CLASSIFY": HOTSPOT_CLASSIFY,
}
NetworkIDs = list(NetworkDictionary.keys())

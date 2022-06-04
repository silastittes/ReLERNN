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
    genotype_inputs,
    position_inputs,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    position_units,
    dropout=0,
    mlp_dropout=0,

):
    x = genotype_inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    #weights for position data (indexed in case additional layers might be stacked later)
    x2 = layers.Dense(position_units[0])(position_inputs)

    #combined weights
    x = layers.concatenate([x, x2])
    x = layers.Dense(64)(x)
    x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return Model([genotype_inputs, position_inputs], outputs)



def TRANSFORMER_PILOT(x, y, **kwargs):
    """
    Verifying updates, but will eventually be transformer model.
    """


    ##############################
    #### PARAMS FOR TUNNING! #####
    ##############################

    numTransformerBlocks = kwargs.get("numTransformerBlocks", 6)
    numHeads = kwargs.get("numHeads", 2)
    headSize = kwargs.get("headSize", 128)
    ffDim = kwargs.get("ffDim", 4)
    mlpSize = kwargs.get("mlpSize", 128)
    positionSize = kwargs.get("positionSize", 256)
    mlpDropout = kwargs.get("mlpDropout", 0.0)
    dropout = kwargs.get("dropout", 0.0)
    LearningRate = kwargs.get("LearningRate", 0.0001)

    ##############################
    ##############################
    ##############################

    haps, pos = x

    numSNPs = haps[0].shape[0]
    numSamps = haps[0].shape[1]
    numPos = pos[0].shape[0]

    genotype_inputs = layers.Input(shape=(numSNPs, numSamps))
    position_inputs = layers.Input(shape=(numPos,))
    
    model = build_model(
        genotype_inputs=genotype_inputs,
        position_inputs=position_inputs,
        head_size=headSize,
        num_heads=numHeads,
        ff_dim=ffDim,
        num_transformer_blocks=numTransformerBlocks,
        mlp_units=[mlpSize],
        position_units=[positionSize],
        mlp_dropout=mlpDropout,
        dropout=dropout,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LearningRate),
        loss="mse",
    )
    model.summary()

    return model


def GRU_TUNED84(x, y, **kwargs):
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


def GRU_POOLED(x, y, **kwargs):

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


def HOTSPOT_CLASSIFY(x, y, **kwargs):

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

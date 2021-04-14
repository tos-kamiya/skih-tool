def build_nn(params):
    seq_length, vocabulary_size, layers, embedding_dim, upside_dim, downside_dim, lr, dropout = \
        params['seq_length'], params['vocabulary_size'], params['layers'], params['embedding_dim'], params['upside_dim'], params['downside_dim'], params['lr'], params['dropout']

    from tensorflow.keras import Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GRU, Dense, Embedding, concatenate

    embedding = Embedding(input_dim=vocabulary_size, input_length=seq_length, output_dim=embedding_dim, mask_zero=True)
    upsideInput = Input(shape=(seq_length, ), name='upside_inp')
    upside_i = embedding(upsideInput)
    for i in range(layers):
        upside_i = GRU(upside_dim, return_sequences=i < layers - 1, name='upside_%d' % (i + 1), dropout=dropout)(upside_i)
    downsideInput = Input(shape=(seq_length, ), name='downside_inp')
    downside_i = embedding(downsideInput)
    for i in range(layers):
        downside_i = GRU(downside_dim, return_sequences=i < layers - 1, name='downside_%d' % (i + 1), dropout=dropout)(downside_i)
    output = Dense(1, activation='sigmoid')(concatenate([upside_i, downside_i]))
    
    model = Model(
        inputs=[upsideInput, downsideInput],
        outputs=[output]
    )
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])

    return model

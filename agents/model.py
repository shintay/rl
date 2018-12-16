from keras.layers import Dense, Input, Lambda, Add, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        h0 = Dense(units=32, activation='relu')(states)
        h1 = Dense(units=64, activation='relu')(h0)
        h2 = Dense(units=32, activation='relu')(h1)

        #TODO
        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Output layer with sigmoid activation
        raw_actions = Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(h2)

        # Scale [0, 1] output for each action dimension to proper range
        actions = Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create model
        self.model = Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        #TODO
        #Incorporate any additional losses here (e.g. from regularizers)

        optimizer = Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = Input(shape=(self.state_size,), name='states')
        actions = Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        h0_states = Dense(units=32, activation='relu')(states)
        h1_states = Dense(units=64, activation='relu')(h0_states)

        # Add hidden layer(s) for action pathway
        h0_actions = Dense(units=32, activation='relu')(actions)
        h1_actions = Dense(units=64, activation='relu')(h0_actions)

        #TODO
        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        model = Add()([h1_states, h1_actions])
        model = Activation('relu')(model)

        #TODO
        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = Dense(units=1, name='q_values')(model)

        # Create model
        self.model = Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients
        # (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

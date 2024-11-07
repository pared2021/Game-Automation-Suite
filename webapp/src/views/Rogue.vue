<template>
  <div class="rogue">
    <h1>Rogue Mode</h1>
    <div v-if="error" class="error-message">
      {{ error }}
    </div>
    <div class="game-state">
      <p>Level: {{ gameState.level }}</p>
      <p>Health: {{ gameState.health }}</p>
      <p>Gold: {{ gameState.gold }}</p>
    </div>
    <div class="blessings">
      <h2>Active Blessings</h2>
      <ul>
        <li v-for="blessing in gameState.blessings" :key="blessing">{{ blessing }}</li>
      </ul>
    </div>
    <button @click="startNewRun" :disabled="isRunning">Start New Run</button>
    <button @click="advanceLevel" :disabled="!isRunning">Advance Level</button>
    <button @click="takeDamage(10)" :disabled="!isRunning">Take 10 Damage</button>
    <button @click="endRun" :disabled="!isRunning">End Run</button>
  </div>
</template>

<script>
import { mapState, mapActions } from 'vuex'

export default {
  name: 'Rogue',
  computed: {
    ...mapState({
      gameState: state => state.rogue.gameState,
      isRunning: state => state.rogue.isRunning,
      error: state => state.rogue.error
    })
  },
  methods: {
    ...mapActions('rogue', ['startNewRun', 'advanceLevel', 'takeDamage', 'endRun'])
  }
}
</script>

<style scoped>
.rogue {
  background-color: #f0f0f0;
  border-radius: 8px;
  padding: 20px;
}

.game-state {
  display: flex;
  justify-content: space-around;
  margin-bottom: 20px;
}

.blessings {
  margin-bottom: 20px;
}

button {
  margin: 0 10px;
  padding: 10px 20px;
  font-size: 16px;
  cursor: pointer;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.error-message {
  color: red;
  font-weight: bold;
  margin-bottom: 10px;
}
</style>
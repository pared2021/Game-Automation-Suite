<template>
  <div class="home">
    <h1>{{ $t('home.title') }}</h1>
    
    <!-- 状态显示 -->
    <div class="status-panel" v-if="automationStatus">
      <div class="status-indicator" :class="{ active: automationStatus === 'running' }">
        {{ $t('home.status') }}: {{ automationStatus }}
      </div>
    </div>

    <!-- 控制面板 -->
    <div class="control-panel">
      <button 
        class="control-btn start"
        @click="startAutomation"
        :disabled="isLoading || automationStatus === 'running'"
      >
        {{ $t('home.start') }}
      </button>
      <button 
        class="control-btn stop"
        @click="stopAutomation"
        :disabled="isLoading || automationStatus !== 'running'"
      >
        {{ $t('home.stop') }}
      </button>
    </div>

    <!-- 游戏状态概览 -->
    <div class="game-overview" v-if="gameStatus">
      <div class="stats-panel">
        <h3>{{ $t('home.playerStats') }}</h3>
        <div class="stat-item">
          <span>{{ $t('home.health') }}:</span>
          <div class="progress-bar">
            <div :style="{ width: `${(currentHealth / 100) * 100}%` }" class="progress health"></div>
          </div>
          <span>{{ currentHealth }}/100</span>
        </div>
        <div class="stat-item">
          <span>{{ $t('home.mana') }}:</span>
          <div class="progress-bar">
            <div :style="{ width: `${(currentMana / 100) * 100}%` }" class="progress mana"></div>
          </div>
          <span>{{ currentMana }}/100</span>
        </div>
      </div>

      <div class="current-task" v-if="activeTask">
        <h3>{{ $t('home.currentTask') }}</h3>
        <p>{{ activeTask.name }}</p>
        <p>{{ $t('home.progress') }}: {{ activeTask.progress }}%</p>
      </div>
    </div>

    <!-- 错误提示 -->
    <div v-if="hasError" class="error-message">
      {{ errorMessage }}
      <button class="close-btn" @click="clearError">&times;</button>
    </div>

    <!-- 加载提示 -->
    <div v-if="isLoading" class="loading-overlay">
      <div class="spinner"></div>
    </div>
  </div>
</template>

<script>
import { mapState, mapGetters, mapActions } from 'vuex'

export default {
  name: 'Home',
  
  computed: {
    ...mapState({
      automationStatus: state => state.status
    }),
    ...mapGetters([
      'isLoading',
      'hasError',
      'errorMessage'
    ]),
    ...mapGetters('game', [
      'currentHealth',
      'currentMana',
      'activeTask',
      'gameStatus'
    ])
  },

  methods: {
    ...mapActions([
      'startAutomation',
      'stopAutomation',
      'clearError'
    ])
  }
}
</script>

<style scoped>
.home {
  padding: 20px;
  max-width: 800px;
  margin: 0 auto;
}

.status-panel {
  margin: 20px 0;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.status-indicator {
  display: inline-block;
  padding: 8px 16px;
  border-radius: 20px;
  background: #dc3545;
  color: white;
}

.status-indicator.active {
  background: #28a745;
}

.control-panel {
  display: flex;
  gap: 20px;
  justify-content: center;
  margin: 30px 0;
}

.control-btn {
  padding: 12px 24px;
  border: none;
  border-radius: 6px;
  font-size: 16px;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.3s ease;
}

.control-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.start {
  background: #28a745;
  color: white;
}

.start:hover:not(:disabled) {
  background: #218838;
}

.stop {
  background: #dc3545;
  color: white;
}

.stop:hover:not(:disabled) {
  background: #c82333;
}

.game-overview {
  display: grid;
  grid-template-columns: 1fr;
  gap: 20px;
  margin-top: 30px;
}

.stats-panel {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 10px 0;
}

.progress-bar {
  flex-grow: 1;
  height: 20px;
  background: #e9ecef;
  border-radius: 10px;
  overflow: hidden;
}

.progress {
  height: 100%;
  transition: width 0.3s ease;
}

.progress.health {
  background: #28a745;
}

.progress.mana {
  background: #007bff;
}

.current-task {
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.error-message {
  position: fixed;
  bottom: 20px;
  right: 20px;
  background: #dc3545;
  color: white;
  padding: 15px 40px 15px 15px;
  border-radius: 6px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.close-btn {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  background: none;
  border: none;
  color: white;
  font-size: 20px;
  cursor: pointer;
}

.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255,255,255,0.8);
  display: flex;
  justify-content: center;
  align-items: center;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

@media (min-width: 768px) {
  .game-overview {
    grid-template-columns: 1fr 1fr;
  }
}
</style>

<template>
  <div class="game">
    <h1>{{ $t('game.title') }}</h1>

    <!-- 游戏状态面板 -->
    <div class="game-status" v-if="gameStatus">
      <div class="status-grid">
        <!-- 玩家状态 -->
        <div class="status-card player-stats">
          <h2>{{ $t('game.playerStats') }}</h2>
          <div class="stat-bars">
            <div class="stat-bar">
              <span>{{ $t('game.health') }}</span>
              <div class="progress-bar">
                <div :style="{ width: `${(currentHealth / 100) * 100}%` }" class="progress health"></div>
              </div>
              <span>{{ currentHealth }}/100</span>
            </div>
            <div class="stat-bar">
              <span>{{ $t('game.mana') }}</span>
              <div class="progress-bar">
                <div :style="{ width: `${(currentMana / 100) * 100}%` }" class="progress mana"></div>
              </div>
              <span>{{ currentMana }}/100</span>
            </div>
          </div>
        </div>

        <!-- 背包系统 -->
        <div class="status-card inventory">
          <h2>{{ $t('game.inventory') }}</h2>
          <div class="inventory-grid">
            <div 
              v-for="item in inventoryItems" 
              :key="item.id" 
              class="inventory-item"
              @click="useItem(item.id)"
            >
              <img :src="item.icon" :alt="item.name">
              <span class="item-name">{{ item.name }}</span>
              <span class="item-count" v-if="item.count > 1">x{{ item.count }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 当前任务 -->
      <div class="status-card current-task" v-if="activeTask">
        <h2>{{ $t('game.currentTask') }}</h2>
        <div class="task-details">
          <h3>{{ activeTask.name }}</h3>
          <p>{{ activeTask.description }}</p>
          <div class="task-progress">
            <div class="progress-bar">
              <div :style="{ width: `${activeTask.progress}%` }" class="progress task"></div>
            </div>
            <span>{{ activeTask.progress }}%</span>
          </div>
          <button 
            @click="completeTask(activeTask.id)"
            :disabled="activeTask.progress < 100"
            class="complete-task-btn"
          >
            {{ $t('game.completeTask') }}
          </button>
        </div>
      </div>

      <!-- 战斗状态 -->
      <div class="status-card battle-status" v-if="isInBattle">
        <h2>{{ $t('game.battle') }}</h2>
        <div class="battle-controls">
          <button @click="startBattle" class="battle-btn">
            {{ $t('game.startBattle') }}
          </button>
        </div>
      </div>
    </div>

    <!-- 错误提示 -->
    <div v-if="hasError" class="error-message">
      {{ errorMessage }}
      <button class="close-btn" @click="clearError">&times;</button>
    </div>

    <!-- 加载状态 -->
    <div v-if="isLoading" class="loading-overlay">
      <div class="spinner"></div>
    </div>
  </div>
</template>

<script>
import { mapState, mapGetters, mapActions } from 'vuex'

export default {
  name: 'Game',

  computed: {
    ...mapGetters([
      'isLoading',
      'hasError',
      'errorMessage'
    ]),
    ...mapGetters('game', [
      'currentHealth',
      'currentMana',
      'inventoryItems',
      'activeTask',
      'isInBattle',
      'gameStatus'
    ])
  },

  methods: {
    ...mapActions([
      'clearError'
    ]),
    ...mapActions('game', [
      'startBattle',
      'useItem',
      'completeTask'
    ])
  }
}
</script>

<style scoped>
.game {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.game-status {
  margin-top: 20px;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-bottom: 20px;
}

.status-card {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.stat-bars {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.stat-bar {
  display: flex;
  align-items: center;
  gap: 10px;
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

.progress.task {
  background: #42b983;
}

.inventory-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
  gap: 10px;
  margin-top: 15px;
}

.inventory-item {
  position: relative;
  background: #f8f9fa;
  padding: 10px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.inventory-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.inventory-item img {
  width: 40px;
  height: 40px;
  margin-bottom: 5px;
}

.item-name {
  font-size: 12px;
  display: block;
}

.item-count {
  position: absolute;
  bottom: 5px;
  right: 5px;
  background: rgba(0,0,0,0.7);
  color: white;
  padding: 2px 6px;
  border-radius: 10px;
  font-size: 12px;
}

.task-details {
  margin-top: 15px;
}

.task-progress {
  margin: 15px 0;
  display: flex;
  align-items: center;
  gap: 10px;
}

.complete-task-btn {
  width: 100%;
  padding: 10px;
  background: #42b983;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.complete-task-btn:disabled {
  background: #a8d5c2;
  cursor: not-allowed;
}

.battle-controls {
  display: flex;
  justify-content: center;
  margin-top: 15px;
}

.battle-btn {
  padding: 12px 24px;
  background: #dc3545;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.battle-btn:hover {
  background: #c82333;
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
  z-index: 1000;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #42b983;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

@media (max-width: 768px) {
  .status-grid {
    grid-template-columns: 1fr;
  }

  .inventory-grid {
    grid-template-columns: repeat(auto-fill, minmax(60px, 1fr));
  }
}
</style>

<template>
  <div class="game-view">
    <h1 class="page-title">{{ $t('game.title') }}</h1>

    <div class="game-content" v-if="gameStatus">
      <div class="game-grid">
        <!-- 玩家状态卡片 -->
        <div class="game-card">
          <div class="game-card-header">
            {{ $t('game.playerStats') }}
          </div>
          <div class="game-card-content">
            <div class="stat-group">
              <div class="stat-label">
                <i class="fas fa-heart"></i>
                {{ $t('game.health') }}
              </div>
              <div class="status-bar">
                <div 
                  class="status-bar-progress health"
                  :style="{ width: `${(currentHealth / 100) * 100}%` }"
                ></div>
              </div>
              <span class="stat-value">{{ currentHealth }}/100</span>
            </div>

            <div class="stat-group">
              <div class="stat-label">
                <i class="fas fa-fire-alt"></i>
                {{ $t('game.mana') }}
              </div>
              <div class="status-bar">
                <div 
                  class="status-bar-progress mana"
                  :style="{ width: `${(currentMana / 100) * 100}%` }"
                ></div>
              </div>
              <span class="stat-value">{{ currentMana }}/100</span>
            </div>
          </div>
        </div>

        <!-- 背包卡片 -->
        <div class="game-card">
          <div class="game-card-header">
            {{ $t('game.inventory') }}
          </div>
          <div class="game-card-content">
            <div class="inventory-grid">
              <div 
                v-for="item in inventoryItems" 
                :key="item.id" 
                class="item-slot"
                @click="useItem(item.id)"
                :class="{ 'selected': selectedItem === item.id }"
              >
                <img :src="item.icon" :alt="item.name">
                <div class="item-slot-count" v-if="item.count > 1">
                  {{ item.count }}
                </div>
                <div class="item-name">{{ item.name }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 当前任务卡片 -->
      <div class="game-card" v-if="activeTask">
        <div class="game-card-header">
          {{ $t('game.currentTask') }}
        </div>
        <div class="game-card-content">
          <h3 class="task-card-title">{{ activeTask.name }}</h3>
          <p class="task-card-description">{{ activeTask.description }}</p>
          <div class="task-progress">
            <div class="task-progress-bar">
              <div 
                class="task-progress-value"
                :style="{ width: `${activeTask.progress}%` }"
              ></div>
            </div>
            <span class="task-progress-text">{{ activeTask.progress }}%</span>
          </div>
          <button 
            class="primary-button"
            @click="completeTask(activeTask.id)"
            :disabled="activeTask.progress < 100"
          >
            {{ $t('game.completeTask') }}
          </button>
        </div>
      </div>

      <!-- 战斗状态卡片 -->
      <div class="game-card" v-if="isInBattle">
        <div class="game-card-header">
          {{ $t('game.battle') }}
        </div>
        <div class="game-card-content">
          <div class="battle-controls">
            <button @click="startBattle" class="danger-button">
              {{ $t('game.startBattle') }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { mapState, mapGetters, mapActions } from 'vuex'

export default {
  name: 'Game',
  setup() {
    const selectedItem = ref(null)
    return { selectedItem }
  },
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
.game-view {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.page-title {
  color: #ffffff;
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 24px;
}

.game-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
  margin-bottom: 24px;
}

.stat-group {
  margin-bottom: 16px;
}

.stat-label {
  display: flex;
  align-items: center;
  color: #e0e0e0;
  margin-bottom: 8px;
}

.stat-label i {
  margin-right: 8px;
  color: #f5d742;
}

.stat-value {
  color: #e0e0e0;
  font-size: 14px;
  margin-top: 4px;
}

.inventory-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
  gap: 12px;
}

.task-progress {
  margin: 16px 0;
}

.task-progress-text {
  color: #e0e0e0;
  font-size: 14px;
  margin-top: 4px;
}

.battle-controls {
  display: flex;
  justify-content: center;
  margin-top: 16px;
}

@media (max-width: 768px) {
  .game-grid {
    grid-template-columns: 1fr;
  }

  .inventory-grid {
    grid-template-columns: repeat(auto-fill, minmax(60px, 1fr));
  }
}
</style>

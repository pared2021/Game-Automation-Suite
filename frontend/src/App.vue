<template>
  <div id="GameAutomationWindow" :class="{ 'dark-theme': isDarkTheme }">
    <div class="app-container">
      <!-- 导航栏 -->
      <nav class="navigation-interface">
        <div class="nav-menu">
          <div class="nav-group">
            <div class="nav-group-title">{{ $t('nav.main') }}</div>
            <router-link to="/" custom v-slot="{ navigate, isActive }">
              <div @click="navigate" class="nav-item" :class="{ active: isActive }">
                <i class="nav-icon fas fa-home"></i>
                {{ $t('general.home') }}
              </div>
            </router-link>
            <router-link to="/game" custom v-slot="{ navigate, isActive }">
              <div @click="navigate" class="nav-item" :class="{ active: isActive }">
                <i class="nav-icon fas fa-gamepad"></i>
                {{ $t('general.game') }}
              </div>
            </router-link>
            <router-link to="/rogue" custom v-slot="{ navigate, isActive }">
              <div @click="navigate" class="nav-item" :class="{ active: isActive }">
                <i class="nav-icon fas fa-dungeon"></i>
                {{ $t('general.rogue') }}
              </div>
            </router-link>
          </div>

          <div class="nav-group">
            <div class="nav-group-title">{{ $t('nav.settings') }}</div>
            <router-link to="/about" custom v-slot="{ navigate, isActive }">
              <div @click="navigate" class="nav-item" :class="{ active: isActive }">
                <i class="nav-icon fas fa-info-circle"></i>
                {{ $t('general.about') }}
              </div>
            </router-link>
          </div>
        </div>

        <!-- 语言选择器 -->
        <select v-model="$i18n.locale" class="language-selector">
          <option value="en-US">English</option>
          <option value="zh-CN">中文</option>
          <option value="ja-JP">日本語</option>
        </select>

        <!-- 主题切换 -->
        <div class="theme-toggle nav-item" @click="toggleTheme">
          <i class="nav-icon" :class="isDarkTheme ? 'fas fa-sun' : 'fas fa-moon'"></i>
          {{ $t('general.theme') }}
        </div>
      </nav>

      <!-- 主内容区 -->
      <main id="mainContent">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </main>
    </div>

    <!-- 全局加载状态 -->
    <div v-if="isLoading" class="loading-overlay">
      <div class="spinner"></div>
    </div>

    <!-- 全局错误提示 -->
    <div v-if="hasError" class="error-message">
      {{ errorMessage }}
      <button class="close-btn" @click="clearError">&times;</button>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { mapState, mapGetters, mapActions } from 'vuex'

export default {
  name: 'App',
  setup() {
    const isDarkTheme = ref(true)

    const toggleTheme = () => {
      isDarkTheme.value = !isDarkTheme.value
      document.documentElement.setAttribute('data-theme', isDarkTheme.value ? 'dark' : 'light')
    }

    return {
      isDarkTheme,
      toggleTheme
    }
  },
  computed: {
    ...mapGetters([
      'isLoading',
      'hasError',
      'errorMessage'
    ])
  },
  methods: {
    ...mapActions([
      'clearError'
    ])
  },
  mounted() {
    // 初始化时获取游戏状态
    this.$store.dispatch('fetchGameState')
  }
}
</script>

<style>
@import './assets/ui/qss/dark/app_window.qss';
@import './assets/ui/qss/dark/navigation_interface.qss';
@import './assets/ui/qss/dark/game_button.qss';
@import './assets/ui/qss/dark/game_card.qss';

/* 基础样式重置 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-color: rgb(25,29,34);
}

/* 应用容器布局 */
.app-container {
  display: flex;
  min-height: 100vh;
}

#mainContent {
  flex-grow: 1;
  padding: 24px;
  overflow-y: auto;
}

/* 过渡动画 */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

/* 加载状态 */
.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.spinner {
  width: 50px;
  height: 50px;
  border: 5px solid #f3f3f3;
  border-top: 5px solid #f5d742;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* 错误提示 */
.error-message {
  position: fixed;
  bottom: 24px;
  right: 24px;
  background: #dc3545;
  color: white;
  padding: 16px 40px 16px 16px;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 1000;
}

.close-btn {
  position: absolute;
  right: 16px;
  top: 50%;
  transform: translateY(-50%);
  background: none;
  border: none;
  color: white;
  font-size: 20px;
  cursor: pointer;
  padding: 4px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .app-container {
    flex-direction: column;
  }

  .navigation-interface {
    min-width: auto;
    border-right: none;
    border-bottom: 1px solid rgb(55,60,65);
  }

  .nav-menu {
    display: flex;
    overflow-x: auto;
    padding: 8px;
  }

  .nav-group {
    margin: 0 8px;
  }

  .nav-item {
    white-space: nowrap;
  }
}
</style>

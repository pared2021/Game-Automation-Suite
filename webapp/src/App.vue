<template>
  <div id="app">
    <header>
      <nav>
        <router-link to="/">{{ $t('general.home') }}</router-link> |
        <router-link to="/about">{{ $t('general.about') }}</router-link> |
        <router-link to="/rogue">{{ $t('general.rogue') }}</router-link>
      </nav>
      <select v-model="$i18n.locale">
        <option value="en-US">English</option>
        <option value="zh-CN">中文</option>
        <option value="ja-JP">日本語</option>
      </select>
    </header>
    <main>
      <router-view/>
    </main>
    <footer>
      <p>&copy; 2023 Game Automation Project</p>
    </footer>
  </div>
</template>

<script>
export default {
  name: 'App',
  mounted() {
    // 在组件挂载后获取游戏状态
    this.$store.dispatch('fetchGameState')
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

header {
  background-color: #f8f9fa;
  padding: 1rem;
}

nav {
  padding: 1rem 0;
}

nav a {
  font-weight: bold;
  color: #2c3e50;
  text-decoration: none;
  margin: 0 0.5rem;
}

nav a.router-link-exact-active {
  color: #42b983;
}

main {
  flex-grow: 1;
  padding: 2rem;
}

footer {
  background-color: #f8f9fa;
  padding: 1rem;
}

select {
  margin-top: 1rem;
}

@media (max-width: 768px) {
  nav {
    flex-direction: column;
  }

  nav a {
    margin: 0.5rem 0;
  }
}
</style>
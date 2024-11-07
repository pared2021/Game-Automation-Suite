<template>
  <div id="app">
    <header>
      <nav>
        <router-link to="/">{{ $t('general.home') }}</router-link> |
        <router-link to="/about">{{ $t('general.about') }}</router-link> |
        <router-link to="/game">{{ $t('general.game') }}</router-link> |
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
      <p>&copy; 2023 Game Automation Suite</p>
    </footer>
  </div>
</template>

<script>
export default {
  name: 'App',
  mounted() {
    // 初始化时获取游戏状态
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
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

nav {
  padding: 1rem 0;
}

nav a {
  font-weight: bold;
  color: #2c3e50;
  text-decoration: none;
  margin: 0 0.5rem;
  transition: color 0.3s ease;
}

nav a:hover {
  color: #42b983;
}

nav a.router-link-exact-active {
  color: #42b983;
}

main {
  flex-grow: 1;
  padding: 2rem;
  background-color: #ffffff;
}

footer {
  background-color: #f8f9fa;
  padding: 1rem;
  box-shadow: 0 -2px 4px rgba(0,0,0,0.1);
}

select {
  margin-top: 1rem;
  padding: 0.5rem;
  border: 1px solid #ddd;
  border-radius: 4px;
  background-color: white;
}

@media (max-width: 768px) {
  nav {
    flex-direction: column;
  }

  nav a {
    margin: 0.5rem 0;
  }

  header {
    padding: 0.5rem;
  }

  main {
    padding: 1rem;
  }
}
</style>

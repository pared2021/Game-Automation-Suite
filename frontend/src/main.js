import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import i18n from './i18n'

// 创建Vue应用实例
const app = createApp(App)

// 注册全局错误处理
app.config.errorHandler = (err, vm, info) => {
  console.error('Global error:', err)
  console.error('Error info:', info)
  store.commit('setError', err.message)
}

// 注册全局属性
app.config.globalProperties.$isDev = process.env.NODE_ENV === 'development'

// 使用插件
app.use(router)
app.use(store)
app.use(i18n)

// 挂载应用
app.mount('#app')

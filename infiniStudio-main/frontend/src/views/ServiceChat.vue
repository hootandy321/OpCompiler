<template>
  <div style="height: calc(100vh - 112px); display: flex; flex-direction: column">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 20px; border-bottom: 2px solid #f0f0f0">
      <div style="display: flex; align-items: center; gap: 16px">
        <a-button @click="goBack" style="border-radius: 6px">← 返回</a-button>
        <h2 style="margin: 0; font-size: 24px; font-weight: 600; color: #262626; display: flex; align-items: center; gap: 12px">
          <span style="font-size: 28px">💬</span>
          <span>{{ serviceName }}</span>
        </h2>
      </div>
      <a-button @click="clearHistory" style="border-radius: 6px">清空历史</a-button>
    </div>

    <div 
      ref="chatContainer" 
      style="flex: 1; overflow-y: auto; padding: 24px; background: #f5f5f5"
    >
      <div v-for="message in messages" :key="message.id" style="margin-bottom: 16px">
        <div :style="{ 
          textAlign: message.role === 'user' ? 'right' : 'left',
          marginBottom: '8px'
        }">
          <a-card 
            :style="{ 
              display: 'inline-block',
              maxWidth: '70%',
              background: message.role === 'user' ? '#1890ff' : '#fff',
              color: message.role === 'user' ? '#fff' : '#000'
            }"
          >
            <div style="white-space: pre-wrap">{{ message.content }}</div>
            <div :style="{ 
              fontSize: '12px', 
              marginTop: '8px',
              opacity: 0.7 
            }">
              {{ formatTime(message.created_at) }}
            </div>
          </a-card>
        </div>
      </div>
    </div>

    <div style="padding: 16px; background: #fff; border-top: 1px solid #e8e8e8">
      <a-input-search
        v-model:value="inputMessage"
        placeholder="输入消息..."
        enter-button="发送"
        size="large"
        @search="sendMessage"
        :loading="sending"
      />
    </div>
  </div>
</template>

<script>
import { ref, onMounted, nextTick, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { getChatHistory, addChatMessage, clearChatHistory, getServices, chatCompletions } from '../api'
import { message } from 'ant-design-vue'

export default {
  name: 'ServiceChat',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const serviceId = parseInt(route.params.id)
    const serviceName = ref('')
    const messages = ref([])
    const inputMessage = ref('')
    const sending = ref(false)
    const chatContainer = ref(null)

    const loadService = async () => {
      try {
        const res = await getServices()
        const service = res.data.find(s => s.id === serviceId)
        if (service) {
          serviceName.value = service.name
        }
      } catch (error) {
        message.error('加载服务信息失败')
      }
    }

    const loadMessages = async () => {
      try {
        const res = await getChatHistory(serviceId)
        messages.value = res.data
        scrollToBottom()
      } catch (error) {
        message.error('加载聊天记录失败')
      }
    }

    const sendMessage = async () => {
      if (!inputMessage.value.trim() || sending.value) {
        return
      }

      const userMessage = inputMessage.value.trim()
      inputMessage.value = ''
      sending.value = true

      try {
        // 添加用户消息到数据库
        await addChatMessage(serviceId, {
          role: 'user',
          content: userMessage
        })

        // 准备消息历史（转换为API格式）
        const apiMessages = messages.value.map(msg => ({
          role: msg.role,
          content: msg.content
        }))
        apiMessages.push({
          role: 'user',
          content: userMessage
        })

        // 调用大模型API（流式响应）
        const requestData = {
          model: 'jiuge',
          messages: apiMessages,
          temperature: 1.0,
          top_k: 50,
          top_p: 0.8,
          max_tokens: 512,
          stream: true
        }

        // 创建临时的助手消息用于显示流式响应
        const tempAssistantMessage = {
          id: 'temp-' + Date.now(),
          role: 'assistant',
          content: '',
          created_at: new Date().toISOString()
        }
        messages.value.push(tempAssistantMessage)
        scrollToBottom()

        // 处理流式响应
        let fullResponse = ''
        try {
          const response = await fetch(`/api/services/${serviceId}/chat/completions`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
          })

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`)
          }

          const reader = response.body.getReader()
          const decoder = new TextDecoder()

          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            const chunk = decoder.decode(value, { stream: true })
            const lines = chunk.split('\n')

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6)
                if (data === '[DONE]') {
                  continue
                }
                try {
                  const json = JSON.parse(data)
                  if (json.choices && json.choices.length > 0) {
                    const delta = json.choices[0].delta
                    if (delta && delta.content) {
                      fullResponse += delta.content
                      // 更新临时消息内容
                      const tempMsg = messages.value.find(m => m.id === tempAssistantMessage.id)
                      if (tempMsg) {
                        tempMsg.content = fullResponse
                        scrollToBottom()
                      }
                    }
                  }
                } catch (e) {
                  // 忽略解析错误
                }
              }
            }
          }

          // 流式响应完成，保存到数据库
          if (fullResponse) {
            await addChatMessage(serviceId, {
              role: 'assistant',
              content: fullResponse
            })
          }

          // 移除临时消息，重新加载消息列表
          messages.value = messages.value.filter(m => m.id !== tempAssistantMessage.id)
          await loadMessages()
        } catch (error) {
          // 移除临时消息
          messages.value = messages.value.filter(m => m.id !== tempAssistantMessage.id)
          throw error
        }
      } catch (error) {
        console.error('发送消息失败:', error)
        message.error('发送消息失败: ' + (error.response?.data?.error || error.message))
      } finally {
        sending.value = false
      }
    }

    const clearHistory = async () => {
      try {
        await clearChatHistory(serviceId)
        messages.value = []
        message.success('历史记录已清空')
      } catch (error) {
        message.error('清空历史记录失败')
      }
    }

    const formatTime = (timeStr) => {
      if (!timeStr) return ''
      const date = new Date(timeStr)
      return date.toLocaleString('zh-CN')
    }

    const scrollToBottom = () => {
      nextTick(() => {
        if (chatContainer.value) {
          chatContainer.value.scrollTop = chatContainer.value.scrollHeight
        }
      })
    }

    watch(messages, () => {
      scrollToBottom()
    }, { deep: true })

    const goBack = () => {
      router.push({ name: 'services' })
    }

    onMounted(() => {
      loadService()
      loadMessages()
    })

    return {
      serviceName,
      messages,
      inputMessage,
      sending,
      chatContainer,
      sendMessage,
      clearHistory,
      formatTime,
      goBack
    }
  }
}
</script>


<template>
  <div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px">
      <h2 style="margin: 0; font-size: 24px; font-weight: 600; color: #262626; display: flex; align-items: center; gap: 12px">
        <span style="font-size: 28px">⏰</span>
        <span>计划任务</span>
      </h2>
      <a-button type="primary" @click="showModal" size="large" style="height: 40px; padding: 0 24px; font-weight: 500">
        <span style="margin-right: 8px">➕</span>添加任务
      </a-button>
    </div>

    <a-card :bordered="false">
      <a-table 
        :columns="columns" 
        :data-source="tasks" 
        :pagination="false"
        row-key="id"
        :bordered="false"
      >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'status'">
          <a-tag :color="getStatusColor(record.status)" style="font-weight: 500; padding: 4px 12px; border-radius: 4px">
            {{ getStatusText(record.status) }}
          </a-tag>
        </template>
        <template v-if="column.key === 'last_run'">
          <span style="color: #595959; font-size: 14px">
            {{ record.last_run ? new Date(record.last_run).toLocaleString('zh-CN', {
              year: 'numeric',
              month: '2-digit',
              day: '2-digit',
              hour: '2-digit',
              minute: '2-digit'
            }) : '未执行' }}
          </span>
        </template>
        <template v-if="column.key === 'next_run'">
          <span style="color: #595959; font-size: 14px">{{ getNextRunTime(record) }}</span>
        </template>
        <template v-if="column.key === 'action'">
          <a-button type="link" @click="executeTask(record)">重新执行</a-button>
          <a-button type="link" @click="viewResult(record)">查看结果</a-button>
          <a-button type="link" @click="editTask(record)">编辑</a-button>
          <a-popconfirm title="确定删除这个任务吗？" @confirm="deleteTask(record.id)">
            <a-button type="link" danger>删除</a-button>
          </a-popconfirm>
        </template>
      </template>
    </a-table>
    </a-card>

    <a-modal
      v-model:open="modalVisible"
      :title="editingTask ? '编辑任务' : '添加任务'"
      width="600px"
      @ok="handleSubmit"
      @cancel="handleCancel"
    >
      <a-form :model="form" :label-col="{ span: 6 }" :wrapper-col="{ span: 18 }">
        <a-form-item label="任务名称" required>
          <a-input v-model:value="form.name" placeholder="请输入任务名称" />
        </a-form-item>
        <a-form-item label="任务命令" required>
          <a-textarea v-model:value="form.command" :rows="4" placeholder="请输入要执行的命令" />
        </a-form-item>
        <a-form-item label="执行服务器" required>
          <a-select v-model:value="form.server_id" placeholder="请选择服务器">
            <a-select-option v-for="server in servers" :key="server.id" :value="server.id">
              {{ server.name }} ({{ server.host_ip }})
            </a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="执行周期" required>
          <a-select v-model:value="form.schedule_type" placeholder="请选择执行周期">
            <a-select-option value="once">一次</a-select-option>
            <a-select-option value="daily">每天</a-select-option>
            <a-select-option value="weekly">每周</a-select-option>
          </a-select>
        </a-form-item>
      </a-form>
    </a-modal>

    <a-modal
      v-model:open="resultModalVisible"
      title="任务执行结果"
      width="800px"
      :footer="null"
    >
      <pre style="background: #f5f5f5; padding: 16px; border-radius: 4px; max-height: 500px; overflow: auto">{{ taskResult }}</pre>
    </a-modal>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { getTasks, getServers, createTask, updateTask, deleteTask as deleteTaskApi, executeTask as executeTaskApi, getTaskResult } from '../api'
import { message } from 'ant-design-vue'

export default {
  name: 'Tasks',
  setup() {
    const tasks = ref([])
    const servers = ref([])
    const modalVisible = ref(false)
    const resultModalVisible = ref(false)
    const editingTask = ref(null)
    const taskResult = ref('')
    const form = ref({
      name: '',
      command: '',
      server_id: null,
      schedule_type: 'once'
    })

    const columns = [
      { title: '任务名称', dataIndex: 'name', key: 'name' },
      { title: '任务命令', dataIndex: 'command', key: 'command' },
      { title: '执行服务器', dataIndex: 'server_name', key: 'server_name' },
      { title: '状态', key: 'status' },
      { title: '最近执行时间', key: 'last_run' },
      { title: '下次执行时间', key: 'next_run' },
      { title: '操作', key: 'action' }
    ]

    const getStatusColor = (status) => {
      const colors = {
        'pending': 'default',
        'executing': 'processing',
        'completed': 'success',
        'failed': 'error'
      }
      return colors[status] || 'default'
    }

    const getStatusText = (status) => {
      const texts = {
        'pending': '待执行',
        'executing': '执行中',
        'completed': '已完成',
        'failed': '执行失败'
      }
      return texts[status] || status
    }

    const getNextRunTime = (task) => {
      if (task.schedule_type === 'once') {
        if (task.last_run) {
          return '已完成（不再次执行）'
        } else {
          return '待执行'
        }
      }
      
      const now = new Date()
      let nextRun
      
      if (task.last_run) {
        // 如果已经执行过，基于last_run计算下次执行时间
        const lastRun = new Date(task.last_run)
        if (task.schedule_type === 'daily') {
          nextRun = new Date(lastRun)
          nextRun.setDate(nextRun.getDate() + 1)
          nextRun.setHours(0, 0, 0, 0)
        } else if (task.schedule_type === 'weekly') {
          nextRun = new Date(lastRun)
          nextRun.setDate(nextRun.getDate() + 7)
          nextRun.setHours(0, 0, 0, 0)
        }
      } else {
        // 如果还没有执行过，根据schedule_type设置下次执行时间
        if (task.schedule_type === 'daily') {
          nextRun = new Date(now)
          nextRun.setHours(0, 0, 0, 0) // 设置为今天0点
          if (nextRun <= now) {
            nextRun.setDate(nextRun.getDate() + 1) // 如果今天0点已过，设置为明天0点
          }
        } else if (task.schedule_type === 'weekly') {
          nextRun = new Date(now)
          nextRun.setDate(nextRun.getDate() + 7) // 设置为7天后
          nextRun.setHours(0, 0, 0, 0)
        }
      }
      
      if (nextRun) {
        // 如果计算出的时间已经过去，说明需要再等一个周期
        if (nextRun <= now && task.schedule_type === 'daily') {
          nextRun.setDate(nextRun.getDate() + 1)
        } else if (nextRun <= now && task.schedule_type === 'weekly') {
          nextRun.setDate(nextRun.getDate() + 7)
        }
        
        return nextRun.toLocaleString('zh-CN', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit'
        })
      }
      
      return '待执行'
    }

    const loadData = async () => {
      try {
        const [tasksRes, serversRes] = await Promise.all([
          getTasks(),
          getServers()
        ])
        tasks.value = tasksRes.data
        servers.value = serversRes.data
      } catch (error) {
        message.error('加载数据失败')
      }
    }

    const showModal = () => {
      editingTask.value = null
      form.value = {
        name: '',
        command: '',
        server_id: null,
        schedule_type: 'once'
      }
      modalVisible.value = true
    }

    const editTask = (task) => {
      editingTask.value = task
      form.value = {
        name: task.name,
        command: task.command,
        server_id: task.server_id,
        schedule_type: task.schedule_type
      }
      modalVisible.value = true
    }

    const handleSubmit = async () => {
      if (!form.value.name || !form.value.command || !form.value.server_id) {
        message.warning('请填写必填项')
        return
      }
      
      try {
        if (editingTask.value) {
          await updateTask(editingTask.value.id, form.value)
          message.success('更新成功')
        } else {
          await createTask(form.value)
          message.success('创建成功')
        }
        modalVisible.value = false
        loadData()
      } catch (error) {
        message.error('操作失败')
      }
    }

    const handleCancel = () => {
      modalVisible.value = false
    }

    const deleteTask = async (id) => {
      try {
        await deleteTaskApi(id)
        message.success('删除成功')
        loadData()
      } catch (error) {
        message.error('删除失败')
      }
    }

    const executeTask = async (task) => {
      try {
        message.loading({ content: '正在执行任务...', key: 'executing' })
        const res = await executeTaskApi(task.id)
        message.success({ content: '任务执行完成', key: 'executing' })
        loadData()
        if (res.data.output) {
          taskResult.value = res.data.output
          if (res.data.error) {
            taskResult.value += '\n\n错误信息:\n' + res.data.error
          }
          resultModalVisible.value = true
        }
      } catch (error) {
        message.error({ content: '任务执行失败', key: 'executing' })
      }
    }

    const viewResult = async (task) => {
      try {
        const res = await getTaskResult(task.id)
        let result = ''
        
        // 添加执行时间信息
        if (res.data.last_run) {
          const executeTime = new Date(res.data.last_run).toLocaleString('zh-CN')
          result = `执行时间: ${executeTime}\n\n`
        }
        
        if (res.data.output !== undefined || res.data.error !== undefined) {
          // 新格式：从数据库读取的结果
          if (res.data.output) {
            result += res.data.output
          }
          if (res.data.error) {
            result += (res.data.output ? '\n\n错误信息:\n' : '错误信息:\n') + res.data.error
          }
          if (!res.data.output && !res.data.error) {
            result += '暂无输出'
          }
        } else if (res.data.message) {
          // 旧格式或没有结果的情况
          result += res.data.message
        } else {
          result += '暂无结果'
        }
        
        taskResult.value = result
        resultModalVisible.value = true
      } catch (error) {
        message.error('获取结果失败')
      }
    }

    onMounted(() => {
      loadData()
    })

    return {
      tasks,
      servers,
      columns,
      modalVisible,
      resultModalVisible,
      editingTask,
      taskResult,
      form,
      getStatusColor,
      getStatusText,
      getNextRunTime,
      showModal,
      editTask,
      handleSubmit,
      handleCancel,
      deleteTask,
      executeTask,
      viewResult
    }
  }
}
</script>


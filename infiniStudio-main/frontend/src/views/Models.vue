<template>
  <div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px">
      <h2 style="margin: 0; font-size: 24px; font-weight: 600; color: #262626; display: flex; align-items: center; gap: 12px">
        <span style="font-size: 28px">🤖</span>
        <span>模型管理</span>
      </h2>
      <a-button type="primary" @click="showModal" size="large" style="height: 40px; padding: 0 24px; font-weight: 500">
        <span style="margin-right: 8px">➕</span>添加模型
      </a-button>
    </div>

    <a-row :gutter="[16, 16]">
      <a-col :xs="24" :sm="12" :md="8" :lg="6" v-for="model in models" :key="model.id">
        <a-card 
          hoverable
          :bordered="false"
          style="border: 1px solid #f0f0f0; border-radius: 12px; overflow: hidden"
        >
          <template #cover v-if="model.logo">
            <div style="background: #fafafa; padding: 24px; display: flex; align-items: center; justify-content: center; min-height: 180px">
              <img :src="model.logo" style="max-height: 150px; max-width: 100%; object-fit: contain" />
            </div>
          </template>
          <template #cover v-else>
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 48px 24px; display: flex; align-items: center; justify-content: center; min-height: 180px">
              <span style="font-size: 64px; opacity: 0.8">🤖</span>
            </div>
          </template>
          <a-card-meta :title="model.name" :title-style="{ fontSize: '16px', fontWeight: 600, textAlign: 'center', padding: '16px 0 8px 0' }">
            <template #description>
              <div v-if="model.parameters" style="color: #8c8c8c; font-size: 13px; margin-top: 4px">
                参数量：{{ model.parameters }}
              </div>
            </template>
          </a-card-meta>
          <template #actions>
            <a-button type="link" @click="editModel(model)" style="color: #667eea; font-weight: 500">编辑</a-button>
            <a-popconfirm title="确定删除这个模型吗？" @confirm="deleteModel(model.id)">
              <a-button type="link" danger style="font-weight: 500">删除</a-button>
            </a-popconfirm>
          </template>
        </a-card>
      </a-col>
    </a-row>

    <a-modal
      v-model:open="modalVisible"
      :title="editingModel ? '编辑模型' : '添加模型'"
      @ok="handleSubmit"
      @cancel="handleCancel"
    >
      <a-form :model="form" :label-col="{ span: 6 }" :wrapper-col="{ span: 18 }">
        <a-form-item label="模型名称" required>
          <a-input v-model:value="form.name" placeholder="请输入模型名称" />
        </a-form-item>
        <a-form-item label="模型参数量">
          <a-input v-model:value="form.parameters" placeholder="例如：7B、13B、70B等" />
        </a-form-item>
        <a-form-item label="Logo">
          <a-upload
            v-model:file-list="fileList"
            name="logo"
            list-type="picture-card"
            :max-count="1"
            :before-upload="beforeUpload"
            @preview="handlePreview"
            @remove="handleRemove"
          >
            <div v-if="fileList.length < 1">
              <plus-outlined />
              <div style="margin-top: 8px">上传</div>
            </div>
          </a-upload>
          <div v-if="form.logo && !fileList.length" style="margin-top: 8px">
            <img :src="form.logo" alt="logo" style="max-width: 200px; max-height: 200px" />
            <div style="margin-top: 8px">
              <a-button size="small" @click="clearLogo">清除Logo</a-button>
            </div>
          </div>
        </a-form-item>
      </a-form>
    </a-modal>

    <a-modal v-model:open="previewVisible" :footer="null">
      <img alt="preview" style="width: 100%" :src="previewImage" />
    </a-modal>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { getModels, createModel, updateModel, deleteModel as deleteModelApi } from '../api'
import { message } from 'ant-design-vue'
import { PlusOutlined } from '@ant-design/icons-vue'

export default {
  name: 'Models',
  components: {
    PlusOutlined
  },
  setup() {
    const models = ref([])
    const modalVisible = ref(false)
    const editingModel = ref(null)
    const form = ref({ name: '', logo: '', parameters: '' })
    const fileList = ref([])
    const previewVisible = ref(false)
    const previewImage = ref('')

    const loadModels = async () => {
      try {
        const res = await getModels()
        models.value = res.data
      } catch (error) {
        message.error('加载模型列表失败')
      }
    }

    const getBase64 = (file) => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.readAsDataURL(file)
        reader.onload = () => resolve(reader.result)
        reader.onerror = error => reject(error)
      })
    }

    const beforeUpload = (file) => {
      const isImage = file.type.startsWith('image/')
      if (!isImage) {
        message.error('只能上传图片文件!')
        return false
      }
      const isLt2M = file.size / 1024 / 1024 < 2
      if (!isLt2M) {
        message.error('图片大小不能超过 2MB!')
        return false
      }
      return false // 阻止自动上传，手动处理
    }

    const handlePreview = async (file) => {
      if (!file.url && !file.preview) {
        file.preview = await getBase64(file.originFileObj)
      }
      previewImage.value = file.url || file.preview
      previewVisible.value = true
    }

    const handleRemove = () => {
      fileList.value = []
    }

    const clearLogo = () => {
      form.value.logo = ''
    }

    const showModal = () => {
      editingModel.value = null
      form.value = { name: '', logo: '', parameters: '' }
      fileList.value = []
      modalVisible.value = true
    }

    const editModel = (model) => {
      editingModel.value = model
      form.value = { name: model.name, logo: model.logo || '', parameters: model.parameters || '' }
      fileList.value = []
      modalVisible.value = true
    }

    const handleSubmit = async () => {
      if (!form.value.name) {
        message.warning('请输入模型名称')
        return
      }
      
      try {
        const file = fileList.value.length > 0 ? fileList.value[0].originFileObj : null
        if (editingModel.value) {
          await updateModel(editingModel.value.id, form.value, file)
          message.success('更新成功')
        } else {
          await createModel(form.value, file)
          message.success('创建成功')
        }
        modalVisible.value = false
        fileList.value = []
        loadModels()
      } catch (error) {
        message.error('操作失败')
      }
    }

    const handleCancel = () => {
      modalVisible.value = false
      form.value = { name: '', logo: '', parameters: '' }
      fileList.value = []
    }

    const deleteModel = async (id) => {
      try {
        await deleteModelApi(id)
        message.success('删除成功')
        loadModels()
      } catch (error) {
        message.error('删除失败')
      }
    }

    onMounted(() => {
      loadModels()
    })

    return {
      models,
      modalVisible,
      editingModel,
      form,
      fileList,
      previewVisible,
      previewImage,
      showModal,
      editModel,
      handleSubmit,
      handleCancel,
      deleteModel,
      beforeUpload,
      handlePreview,
      handleRemove,
      clearLogo
    }
  }
}
</script>


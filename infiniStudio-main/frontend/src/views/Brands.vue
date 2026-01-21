<template>
  <div>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px">
      <h2 style="margin: 0; font-size: 24px; font-weight: 600; color: #262626; display: flex; align-items: center; gap: 12px">
        <span style="font-size: 28px">🏢</span>
        <span>品牌管理</span>
      </h2>
      <a-button type="primary" @click="showModal" size="large" style="height: 40px; padding: 0 24px; font-weight: 500">
        <span style="margin-right: 8px">➕</span>添加品牌
      </a-button>
    </div>

    <draggable
      v-model="brands"
      :animation="200"
      handle=".drag-handle"
      item-key="id"
      @end="handleDragEnd"
      tag="div"
      class="brands-grid"
    >
      <template #item="{ element: brand }">
        <div class="brand-card-wrapper">
          <a-card 
            hoverable
            :bordered="false"
            style="border: 1px solid #f0f0f0; border-radius: 12px; overflow: hidden; position: relative"
            @click="goToBrandDetail(brand.id)"
            :style="{ cursor: 'pointer' }"
          >
            <div class="drag-handle" style="position: absolute; top: 8px; right: 8px; cursor: grab; z-index: 10; padding: 4px; background: rgba(255,255,255,0.9); border-radius: 4px; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" @click.stop @mousedown.stop>
              <span style="font-size: 16px; color: #8c8c8c; user-select: none;">⋮⋮</span>
            </div>
            <template #cover v-if="brand.logo">
              <div style="background: #fafafa; padding: 24px; display: flex; align-items: center; justify-content: center; min-height: 180px">
                <img :src="brand.logo" style="max-height: 150px; max-width: 100%; object-fit: contain" />
              </div>
            </template>
            <template #cover v-else>
              <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 48px 24px; display: flex; align-items: center; justify-content: center; min-height: 180px">
                <span style="font-size: 64px; opacity: 0.8">🏢</span>
              </div>
            </template>
            <a-card-meta :title="brand.name" :title-style="{ fontSize: '16px', fontWeight: 600, textAlign: 'center', padding: '16px 0 8px 0' }" />
            <template #actions>
              <a-button type="link" @click.stop="editBrand(brand)" style="color: #667eea; font-weight: 500">编辑</a-button>
              <a-popconfirm title="确定删除这个品牌吗？" @confirm="deleteBrand(brand.id)">
                <a-button type="link" danger @click.stop style="font-weight: 500">删除</a-button>
              </a-popconfirm>
            </template>
          </a-card>
        </div>
      </template>
    </draggable>

    <a-modal
      v-model:open="modalVisible"
      :title="editingBrand ? '编辑品牌' : '添加品牌'"
      @ok="handleSubmit"
      @cancel="handleCancel"
    >
      <a-form :model="form" :label-col="{ span: 6 }" :wrapper-col="{ span: 18 }">
        <a-form-item label="品牌名称" required>
          <a-input v-model:value="form.name" placeholder="请输入品牌名称" />
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
import { useRouter } from 'vue-router'
import { getBrands, createBrand, updateBrand, deleteBrand as deleteBrandApi, reorderBrands } from '../api'
import { message } from 'ant-design-vue'
import { PlusOutlined } from '@ant-design/icons-vue'
import draggable from 'vuedraggable'

export default {
  name: 'Brands',
  components: {
    PlusOutlined,
    draggable
  },
  setup() {
    const router = useRouter()
    const brands = ref([])
    const modalVisible = ref(false)
    const editingBrand = ref(null)
    const form = ref({ name: '', logo: '' })
    const fileList = ref([])
    const previewVisible = ref(false)
    const previewImage = ref('')

    const loadBrands = async () => {
      try {
        const res = await getBrands()
        brands.value = res.data
      } catch (error) {
        message.error('加载品牌列表失败')
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
      editingBrand.value = null
      form.value = { name: '', logo: '' }
      fileList.value = []
      modalVisible.value = true
    }

    const editBrand = (brand) => {
      editingBrand.value = brand
      form.value = { name: brand.name, logo: brand.logo || '' }
      fileList.value = []
      modalVisible.value = true
    }

    const handleSubmit = async () => {
      if (!form.value.name) {
        message.warning('请输入品牌名称')
        return
      }
      
      try {
        const file = fileList.value.length > 0 ? fileList.value[0].originFileObj : null
        if (editingBrand.value) {
          await updateBrand(editingBrand.value.id, form.value, file)
          message.success('更新成功')
        } else {
          await createBrand(form.value, file)
          message.success('创建成功')
        }
        modalVisible.value = false
        fileList.value = []
        loadBrands()
      } catch (error) {
        message.error('操作失败')
      }
    }

    const handleCancel = () => {
      modalVisible.value = false
      form.value = { name: '', logo: '' }
      fileList.value = []
    }

    const deleteBrand = async (id) => {
      try {
        await deleteBrandApi(id)
        message.success('删除成功')
        loadBrands()
      } catch (error) {
        message.error('删除失败')
      }
    }

    const goToBrandDetail = (brandId) => {
      router.push({ name: 'brandDetail', params: { id: brandId } })
    }

    const handleDragEnd = async () => {
      try {
        // 构建排序数据
        const orders = brands.value.map((brand, index) => ({
          id: brand.id,
          sort_order: index
        }))
        await reorderBrands(orders)
        message.success('排序已保存')
      } catch (error) {
        message.error('保存排序失败')
        // 重新加载数据以恢复原始顺序
        loadBrands()
      }
    }

    onMounted(() => {
      loadBrands()
    })

    return {
      brands,
      modalVisible,
      editingBrand,
      form,
      fileList,
      previewVisible,
      previewImage,
      showModal,
      editBrand,
      handleSubmit,
      handleCancel,
      deleteBrand,
      goToBrandDetail,
      handleDragEnd,
      beforeUpload,
      handlePreview,
      handleRemove,
      clearLogo
    }
  }
}
</script>

<style scoped>
.brands-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 16px;
}

@media (max-width: 576px) {
  .brands-grid {
    grid-template-columns: 1fr;
  }
}

@media (min-width: 577px) and (max-width: 768px) {
  .brands-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (min-width: 769px) and (max-width: 992px) {
  .brands-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (min-width: 993px) {
  .brands-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}

.brand-card-wrapper {
  width: 100%;
}
</style>


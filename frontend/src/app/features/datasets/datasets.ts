import { Component, inject, OnInit, signal, computed } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTableModule } from '@angular/material/table';
import { MatChipsModule } from '@angular/material/chips';
import { DatePipe } from '@angular/common';
import { HttpEventType } from '@angular/common/http';
import { DatasetsService } from '../../core/services/datasets.service';
import { StudiesService } from '../../core/services/studies.service';
import { NotificationService } from '../../core/services/notification.service';
import { AuthService } from '../../core/services/auth.service';
import { Study, Dataset } from '../../core/models';

@Component({
  selector: 'app-datasets',
  imports: [
    ReactiveFormsModule, DatePipe,
    MatCardModule, MatButtonModule, MatIconModule,
    MatFormFieldModule, MatSelectModule, MatProgressBarModule,
    MatProgressSpinnerModule, MatTableModule, MatChipsModule,
  ],
  templateUrl: './datasets.html',
  styleUrl: './datasets.css',
})
export class DatasetsComponent implements OnInit {
  private readonly svc        = inject(DatasetsService);
  private readonly studiesSvc = inject(StudiesService);
  private readonly notify     = inject(NotificationService);
  private readonly route      = inject(ActivatedRoute);
  private readonly fb         = inject(FormBuilder);
  readonly auth               = inject(AuthService);

  readonly studies      = signal<Study[]>([]);
  readonly datasets     = signal<Dataset[]>([]);
  readonly loading      = signal(false);
  readonly uploading    = signal(false);
  readonly uploadProgress = signal(0);
  readonly dragOver     = signal(false);
  readonly selectedFile = signal<File | null>(null);

  readonly columns = ['filename', 'domain', 'file_format', 'row_count', 'study_id', 'created_at'];

  readonly domains = ['DM', 'VS', 'AE', 'CM', 'EX', 'LB', 'DS', 'EG', 'MULTI', 'DATASET_JSON'];

  readonly uploadForm = this.fb.nonNullable.group({
    study_id: ['', Validators.required],
    domain:   ['DM', Validators.required],
  });

  ngOnInit(): void {
    this.studiesSvc.list().subscribe(studies => {
      this.studies.set(studies.filter(s => s.status !== 'archived'));
      // Pre-select study from query param
      const sid = this.route.snapshot.queryParamMap.get('study_id');
      if (sid) this.uploadForm.patchValue({ study_id: sid });
    });
    this.loadDatasets();
  }

  loadDatasets(): void {
    this.loading.set(true);
    const studyId = this.route.snapshot.queryParamMap.get('study_id') ?? '';
    if (studyId) {
      this.svc.list(studyId).subscribe({ next: ds => { this.datasets.set(ds); this.loading.set(false); }, error: () => this.loading.set(false) });
    } else {
      this.loading.set(false);
    }
  }

  onDragOver(e: DragEvent): void { e.preventDefault(); this.dragOver.set(true); }
  onDragLeave(): void { this.dragOver.set(false); }
  onDrop(e: DragEvent): void {
    e.preventDefault();
    this.dragOver.set(false);
    const file = e.dataTransfer?.files[0];
    if (file) this.selectedFile.set(file);
  }
  onFileInput(e: Event): void {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) this.selectedFile.set(file);
  }
  openFilePicker(): void { document.getElementById('fileInput')?.click(); }

  upload(): void {
    if (this.uploadForm.invalid || !this.selectedFile()) return;
    const { study_id, domain } = this.uploadForm.getRawValue();
    this.uploading.set(true);
    this.uploadProgress.set(0);

    this.svc.upload(study_id, domain, this.selectedFile()!).subscribe({
      next: event => {
        if (event.type === HttpEventType.UploadProgress && event.total) {
          this.uploadProgress.set(Math.round(100 * event.loaded / event.total));
        } else if (event.type === HttpEventType.Response) {
          this.uploading.set(false);
          this.selectedFile.set(null);
          this.notify.success('File uploaded successfully.');
          this.svc.list(study_id).subscribe(ds => this.datasets.set(ds));
        }
      },
      error: () => this.uploading.set(false),
    });
  }

  studyName(id: string): string {
    return this.studies().find(s => s.id === id)?.study_id ?? id.slice(0, 8);
  }
}
